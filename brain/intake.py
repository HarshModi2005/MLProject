"""
Intake: turn what the user typed into a campaign they have explicitly verified.

    1. understand   one LLM call: restate the request, pick a playbook, fill the spec, list what was
                    ASSUMED (not said) and the few questions whose answers would change the result
    2. validate     deterministic checks for what a campaign cannot run without (topics, an ask,
                    something true about the sender) — these become questions too
    3. ask          the questions, each with a default the user can accept with Enter
    4. uploads      ask for older lists that would help (playbook-specific suggestions); classify each
                    one (targets / exclude / examples / style / about the sender) and let the user correct it
    5. refine       one LLM call folding in answers, uploads and lookalike examples
    6. verify       show a verification card — understanding, who, where, how many, the facts that
                    emails may use (with their source), lists and assumptions — and loop on edits until
                    the user says yes

Typical cost: 2 LLM calls (+1 per edit, +1 to read a CV or deck, cached by content).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from brain import uploads as up
from brain.context import CampaignContext
from brain.llm import LLM
from brain.models import (Ask, CampaignSpec, Constraints, Contact, FitCriterion, Offer, ProspectProfile, SenderProfile,
                          TargetPersona)
from brain.playbooks import PLAYBOOKS, catalog_text, get_playbook
from brain.searchplan import search_budget
from brain.spec import new_campaign_id, sender_from_text
from brain.textutil import normalise_locations, split_locations, squash

log = logging.getLogger("brain.intake")

_SCHEMA = """{
 "playbook": "<one name from PLAYBOOKS>",
 "playbook_confidence": <0-1: how sure you are this playbook fits; below 0.6 when two playbooks fit about equally>,
 "playbook_alternatives": [{"name": "<another playbook that could fit>", "why": "<one line>"}],
 "understanding": "<2-3 plain sentences restating what the user wants, to whom, and why — as you would read it back to them>",
 "objective": "<the outcome the user wants, one sentence>",
 "ask": "<the concrete request each email makes, one sentence>",
 "ask_detail": {"type": "call|meeting|reply|apply|intro|review|other",
   "link": "<a booking/application/deck link the user gave, or null>", "availability": "<when the sender is free, if stated, or null>",
   "deadline": "<if stated, or null>",
   "inbox_text": "<the ask phrased for a shared team inbox, e.g. 'could you point me to whoever handles sponsorships?'>"},
 "offer": {"value_prop": "<what the sender offers, ONLY if stated, else ''>", "proof_points": ["<customers, numbers, results the user stated>"],
   "terms": "<price/equity/tiers if stated, else ''>", "audience": "<the audience/reach the sender brings, if stated>"},
 "constraints": {"size_min": <employees or null>, "size_max": <employees or null>, "stages": ["<e.g. seed, series a, final-year student>"],
   "seniority": "<if stated>", "remote_ok": <true/false/null>, "radius_km": <number or null>,
   "must": ["<hard requirements the user stated>"], "avoid": ["<kinds of target the user said to avoid>"]},
 "language": "<language the emails should be written in, only if the user said so or the targets clearly need it; '' = English>",
 "target_kind": "person" | "org",
 "persona": {"description": "<who should receive the emails>", "roles": ["<job titles that qualify>"], "org_types": ["<kinds of organisation>"]},
 "topics": ["<2-6 noun phrases a search engine would match for the TARGETS (what they are/do), not for the sender>"],
 "expanded_terms": ["<3-8 synonyms or closely related categories>"],
 "locations": ["<countries or 'City, Country' exactly as the user means them; [] if anywhere>"],
 "organizations": ["<specific organisations the user named as targets>"],
 "exclude_organizations": ["<organisations the user said to avoid>"],
 "timeline": "<when, or null>",
 "tone": "professional" | "friendly" | "concise",
 "target_count": <number of recipients the user wants, or null>,
 "search_queries": ["<6-10 web queries that surface the targets' own websites or curated lists of such targets; never site:linkedin.com>"],
 "lookalike_queries": ["<only if EXAMPLES are given: 2-5 queries to find organisations similar to them>"],
 "criteria": [{"name": "<2-4 things that make a target GOOD beyond the topic, as checkable facts, e.g. 'has sponsored college tech fests', 'hiring backend engineers', 'invests at pre-seed'>",
   "signal": "<one name from SIGNALS: the kind of proof that would show it>",
   "weight": <1-3>, "required": <true only if the user said it is a must>}],
 "anti_patterns": ["<kinds of organisation/person that look relevant but are poor targets, e.g. 'event agency', 'job board'>"],
 "event_queries": ["<only when targets are sponsors or speakers: 2-4 queries for SIMILAR PAST EVENTS' sponsor/speaker pages>"],
 "preferred_inboxes": ["<if targets are organisations: shared inbox names that suit this ask, best first>"],
 "sender_facts": ["<facts about the sender or the offer that the USER stated (numbers, achievements, what is offered), one per item, in their words>"],
 "assumptions": ["<everything you inferred that the user did NOT say: the audience, location, roles, counts...>"],
 "questions": [{"question": "<short question>", "why": "<what changes with the answer>", "default": "<what you will assume if they skip>"}]
}"""

UNDERSTAND_SYSTEM = f"""You configure a cold-outreach campaign from a user's request. Read it carefully and precisely.

PLAYBOOKS (grouped; compare the ones in the same group before choosing):
{{catalog}}

SIGNALS (kinds of proof a criterion can name; discovery adds the sources that prove them):
{{signals}}

Return JSON with exactly these keys:
{_SCHEMA}

Rules:
- Never invent facts about the sender. sender_facts may only contain what the user or their profile states.
- Separate what was said from what you assumed: anything not stated goes into "assumptions".
- Ask at most 4 questions, only ones whose answers would change who is contacted or what the email says.
  Do not ask about things already stated. Give every question a sensible default.
- Topics describe the TARGETS (e.g. "dental clinics", "edtech startups"), not the sender's product.
- offer, constraints, ask_detail.link/availability/deadline and language: only what the user stated; otherwise leave
  them empty (never guess a price, a link or a size).
- criteria are qualities of a GOOD target beyond the topic, each with the SIGNAL that would prove it. Never make a
  criterion of a location (that is "locations") or a job title / role (that is "persona.roles")."""

REFINE_SYSTEM = f"""You update a cold-outreach campaign configuration with new information from the user (answers,
corrections, uploaded lists). Apply corrections exactly; when the user answers a question, remove the corresponding
assumption and question. Keep everything else unchanged unless the new information contradicts it.

PLAYBOOKS:
{{catalog}}

SIGNALS:
{{signals}}

Return the FULL updated JSON with exactly these keys:
{_SCHEMA}
Return an empty "questions" list unless something essential is still unknown."""


# ── UI ───────────────────────────────────────────────────────────────────────

class UI:
    """How the intake talks to the user. ConsoleUI for the terminal; ScriptedUI for tests and APIs."""

    def show(self, text: str) -> None:
        raise NotImplementedError

    def ask(self, prompt: str, default: str = "") -> str:
        raise NotImplementedError


class ConsoleUI(UI):
    def __init__(self):
        from rich.console import Console

        self.console = Console()

    def show(self, text: str) -> None:
        self.console.print(text)

    def ask(self, prompt: str, default: str = "") -> str:
        from rich.prompt import Prompt

        return Prompt.ask(prompt, default=default or "", show_default=bool(default)).strip()


class ScriptedUI(UI):
    def __init__(self, answers: List[str]):
        self.answers = list(answers)
        self.shown: List[str] = []
        self.asked: List[str] = []

    def show(self, text: str) -> None:
        self.shown.append(text)

    def ask(self, prompt: str, default: str = "") -> str:
        self.asked.append(prompt)
        ans = self.answers.pop(0) if self.answers else ""
        return ans if ans else default


# ── result ───────────────────────────────────────────────────────────────────

@dataclass
class IntakeResult:
    spec: CampaignSpec
    ctx: CampaignContext
    seed_contacts: List[Contact] = field(default_factory=list)
    only_seeds: bool = False
    llm_calls: int = 0


# ── steps ────────────────────────────────────────────────────────────────────

def _user_block(prompt: str, sender: SenderProfile, extra: str = "") -> str:
    block = f"USER REQUEST:\n{prompt}\n"
    facts = sender.facts_text()
    if facts.strip():
        block += f"\nSENDER PROFILE (from their CV/description):\n{facts[:2500]}\n"
    return block + extra


def signals_text() -> str:
    """Every registered kind of proof, with the sources that can prove it (brain/discovery/signals.py + manifests)."""
    from brain.discovery.signals import SIGNALS
    from brain.tools import TOOLS

    lines = ["- topic: works on / covers the topics (matched on text)"]
    for sig in SIGNALS.values():
        if not sig.proof:
            continue
        by = [t.name for t in TOOLS.values() if sig.name in t.proves][:4]
        lines.append(f"- {sig.name}: {sig.description}" + (f" (found by: {', '.join(by)})" if by else " (judged from evidence)"))
    return "\n".join(lines)


def _system(template: str) -> str:
    return template.replace("{catalog}", catalog_text()).replace("{signals}", signals_text())


def understand(prompt: str, sender: SenderProfile, llm: LLM) -> Dict[str, Any]:
    data = llm.json(system=_system(UNDERSTAND_SYSTEM), user=_user_block(prompt, sender),
                    purpose="intake", tier="strong", max_tokens=2600)
    return data if isinstance(data, dict) else {}


def refine(data: Dict[str, Any], prompt: str, sender: SenderProfile, llm: LLM, *, answers: List[str] = (),
           edits: List[str] = (), uploads_summary: str = "", seeds: List[str] = ()) -> Dict[str, Any]:
    extra = f"\nCURRENT CONFIGURATION:\n{json.dumps(data, ensure_ascii=False)[:6000]}\n"
    if answers:
        extra += "\nANSWERS TO YOUR QUESTIONS:\n" + "\n".join(f"- {a}" for a in answers) + "\n"
    if edits:
        extra += "\nCORRECTIONS FROM THE USER (apply exactly):\n" + "\n".join(f"- {e}" for e in edits) + "\n"
    if uploads_summary:
        extra += f"\nUPLOADED LISTS:\n{uploads_summary}\n"
    if seeds:
        extra += "\nEXAMPLES THE USER CONSIDERS GOOD TARGETS (write lookalike_queries for them):\n" + ", ".join(seeds[:15]) + "\n"
    out = llm.json(system=_system(REFINE_SYSTEM), user=_user_block(prompt, sender, extra),
                   purpose="intake_refine", tier="strong", max_tokens=2600)
    return out if isinstance(out, dict) and out.get("objective") else data


def validate(data: Dict[str, Any], sender: SenderProfile) -> List[Dict[str, str]]:
    """What the campaign cannot run without, as questions with defaults. No LLM."""
    qs: List[Dict[str, str]] = []
    if not [t for t in data.get("topics") or [] if str(t).strip()]:
        qs.append({"question": "What kind of people or organisations should receive these emails?",
                   "why": "discovery searches for exactly this", "default": ""})
    if not str(data.get("ask") or "").strip():
        qs.append({"question": "What exactly should each email ask for?", "why": "every email needs one clear ask",
                   "default": "a short call"})
    if not (data.get("sender_facts") or sender.highlights or sender.summary):
        qs.append({"question": "Tell me about yourself or your offer in 2-3 sentences (numbers help), or give a "
                               "path to your CV/deck.",
                   "why": "emails may only state facts you provided", "default": ""})
    ask = str(data.get("ask") or "").strip()
    if ask and not _ASK_VERB.search(ask):
        qs.append({"question": f"What exactly should the recipient do? (\"{squash(ask, 60)}\" has no clear action)",
                   "why": "an email without a clear ask rarely gets a reply", "default": f"a short call about {squash(ask, 50)}"})
    mine = _sender_words(sender, data)
    topics = [str(t) for t in data.get("topics") or [] if str(t).strip()]
    persona = str((data.get("persona") or {}).get("description") or "").lower()
    selfish = [t for t in topics if _words(t) and _words(t) <= mine and not (_words(t) & _words(persona))]
    if topics and len(selfish) == len(topics):
        qs.append({"question": f"Are {', '.join(selfish[:3])} what your TARGETS do? They look like your own work — "
                               "who should receive the emails?",
                   "why": "discovery searches for organisations/people matching the topics",
                   "default": ", ".join(topics[:3])})
    try:
        conf = float(data.get("playbook_confidence")) if data.get("playbook_confidence") is not None else 1.0
    except (TypeError, ValueError):
        conf = 1.0
    alts = [a.get("name") for a in data.get("playbook_alternatives") or [] if isinstance(a, dict) and a.get("name") in PLAYBOOKS]
    if conf < 0.6 and alts and data.get("playbook") in PLAYBOOKS:
        a, b = get_playbook(data["playbook"]), get_playbook(alts[0])
        qs.append({"question": f"Is this closer to '{a.title}' or '{b.title}'?",
                   "why": "it decides where to look and what makes a good target", "default": a.title})
    return qs


_ASK_VERB = re.compile(r"\b(call|chat|meet|meeting|talk|reply|respond|apply|introduc|refer|review|share|send|join|"
                       r"sponsor|invest|fund|consider|discuss|try|pilot|book|schedule|connect|feedback|interview|"
                       r"advice|advise|mentor|speak|hire|partner|distribut|buy|quote|demo|visit|collaborat|host|"
                       r"cover|feature|write|guest|stock|list|point|forward|access|internship|position|role|job)", re.I)


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9+-]{3,}", text.lower()) if w not in _STOP}


def _sender_words(sender: SenderProfile, data: Dict[str, Any]) -> set:
    text = " ".join([sender.headline, sender.summary, " ".join(sender.skills)] + list(sender.highlights)
                    + [str(f) for f in data.get("sender_facts") or []])
    return _words(text)


def _merge_questions(data: Dict[str, Any], extra: List[Dict[str, str]], limit: int = 5) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for q in extra + [q for q in (data.get("questions") or []) if isinstance(q, dict)]:
        text = str(q.get("question") or "").strip()
        if text and all(text.lower() != o["question"].lower() for o in out):
            out.append({"question": text, "why": str(q.get("why") or ""), "default": str(q.get("default") or "")})
    return out[:limit]


def build(data: Dict[str, Any], sender: SenderProfile, *, campaign_id: str, target_count: Optional[int],
          locations: Optional[List[str]], playbook: Optional[str]) -> CampaignSpec:
    name = playbook or (data.get("playbook") or data.get("pack"))
    if name == "general" and data.get("pack") in PLAYBOOKS and data.get("pack") != "general":
        name = data.get("pack")
    pb = get_playbook(name if name in PLAYBOOKS else "general")
    locs, remote = normalise_locations(locations if locations else [str(x) for x in data.get("locations") or []])
    countries, regions = split_locations(locs)
    persona = data.get("persona") or {}
    count = target_count or data.get("target_count")
    try:
        count = max(1, min(200, int(count))) if count else 25
    except (TypeError, ValueError):
        count = 25
    kind = data.get("target_kind") if data.get("target_kind") in ("person", "org") else pb.target_kind
    if kind == "person" and _org_only(pb):
        kind = "org"  # this playbook's sources find organisations; the right person is reached through them
    return CampaignSpec(
        profile=_profile(data, persona, pb),
        event_queries=[str(q) for q in data.get("event_queries") or []][:4],
        id=campaign_id, pack=pb.name, target_kind=kind,
        objective=squash(str(data.get("objective") or ""), 300), ask=squash(str(data.get("ask") or ""), 300),
        persona=TargetPersona(description=squash(str(persona.get("description") or ""), 300),
                              roles=[str(r) for r in persona.get("roles") or []][:8] or pb.default_roles[:8],
                              org_types=[str(r) for r in persona.get("org_types") or []][:5]),
        topics=[str(t) for t in data.get("topics") or []][:6],
        expanded_terms=[str(t) for t in data.get("expanded_terms") or []][:8],
        countries=countries, regions=regions,
        organizations=[str(o) for o in data.get("organizations") or []][:15],
        exclude_organizations=[str(o) for o in data.get("exclude_organizations") or []][:15],
        search_queries=[str(q) for q in data.get("search_queries") or []][:10],
        lookalike_queries=[str(q) for q in data.get("lookalike_queries") or []][:5],
        preferred_inboxes=[str(q).lower() for q in data.get("preferred_inboxes") or []][:8] or pb.preferred_inboxes,
        timeline=str(data["timeline"]) if data.get("timeline") else None,
        tone=str(data.get("tone") or "professional"), target_count=count, word_target=pb.word_target, sender=sender,
        ask_detail=_ask(data), offer=_offer(data), constraints=_constraints(data, remote),
        language=_language(data, countries), playbook_confidence=_float(data.get("playbook_confidence")),
        playbook_alternatives=[a["name"] for a in data.get("playbook_alternatives") or []
                               if isinstance(a, dict) and a.get("name") in PLAYBOOKS and a["name"] != pb.name][:2],
    )


def _org_only(pb) -> bool:
    from brain.tools import TOOLS

    kinds = [TOOLS[n].kinds for n in pb.sources if n in TOOLS and TOOLS[n].stage != "expand"]
    return bool(kinds) and all(k == ("org",) for k in kinds)


# Criteria that only restate a place or a role: those have their own fields (locations, persona.roles).
_NOT_A_CRITERION = re.compile(r"\b(based|located|headquartered|operating) in\b|^in [A-Z]|\brole\b|\bjob title\b|"
                              r"\bposition (is|of)\b|^(is|be) an? (journalist|principal|founder|professor|ceo|cto)\b", re.I)


def _float(v) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(v))) if v is not None else None
    except (TypeError, ValueError):
        return None


def _int(v) -> Optional[int]:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _str(v) -> str:
    return "" if v in (None, "null", "None") else squash(str(v), 300)


ASK_TYPES = ("call", "meeting", "reply", "apply", "intro", "review", "other")


def _ask(data: Dict[str, Any]) -> Ask:
    d = data.get("ask_detail") if isinstance(data.get("ask_detail"), dict) else {}
    t = str(d.get("type") or "other").lower()
    link = _str(d.get("link"))
    return Ask(type=t if t in ASK_TYPES else "other", text=squash(str(data.get("ask") or ""), 300),
               link=link if link.startswith("http") else None, availability=_str(d.get("availability")) or None,
               deadline=_str(d.get("deadline")) or None, inbox_text=_str(d.get("inbox_text")))


def _offer(data: Dict[str, Any]) -> Offer:
    d = data.get("offer") if isinstance(data.get("offer"), dict) else {}
    return Offer(value_prop=_str(d.get("value_prop")), terms=_str(d.get("terms")), audience=_str(d.get("audience")),
                 proof_points=[squash(str(p), 200) for p in d.get("proof_points") or [] if str(p).strip()][:6])


def _constraints(data: Dict[str, Any], remote: Optional[bool]) -> Constraints:
    d = data.get("constraints") if isinstance(data.get("constraints"), dict) else {}
    rok = d.get("remote_ok")
    return Constraints(size_min=_int(d.get("size_min")), size_max=_int(d.get("size_max")),
                       stages=[squash(str(x), 40) for x in d.get("stages") or []][:5], seniority=_str(d.get("seniority")),
                       remote_ok=remote if remote is not None else (rok if isinstance(rok, bool) else None),
                       radius_km=_int(d.get("radius_km")),
                       must=[squash(str(x), 100) for x in d.get("must") or []][:5],
                       avoid=[squash(str(x), 60) for x in d.get("avoid") or []][:6])


# Business cold email defaults to the local language in these markets; English elsewhere.
_LOCAL_LANGUAGE = {"DE": "German", "AT": "German", "FR": "French", "ES": "Spanish", "MX": "Spanish", "AR": "Spanish",
                   "CO": "Spanish", "CL": "Spanish", "IT": "Italian", "BR": "Portuguese", "PT": "Portuguese",
                   "JP": "Japanese", "KR": "Korean", "CN": "Chinese"}


def _language(data: Dict[str, Any], countries: List[str]) -> str:
    stated = _str(data.get("language"))
    if stated and stated.lower() not in ("english", "en"):
        return stated
    if stated:
        return ""
    langs = {_LOCAL_LANGUAGE.get(c) for c in countries} if countries else {None}
    return langs.pop() if len(langs) == 1 and None not in langs else ""


def _known_signal(sig: str) -> bool:
    """Any registered kind of proof (brain/discovery/signals.py), not a fixed list."""
    from brain.discovery.signals import known

    return known(sig)
_STOP = {"the", "and", "for", "with", "that", "has", "have", "are", "who", "their", "they", "from", "into", "this",
         "been", "at", "in", "of", "to", "a", "an", "is", "or", "on", "by"}


def _profile(data: Dict[str, Any], persona: Dict[str, Any], pb) -> ProspectProfile:
    """The prospect profile from the intake answer. Criterion terms are the criterion's own content words, so the
    cheap scorer can check them against evidence; the playbook's defaults are merged in later (brain/fit.py)."""
    crits = []
    for c in data.get("criteria") or []:
        if not isinstance(c, dict) or not str(c.get("name") or "").strip():
            continue
        name = squash(str(c["name"]), 120)
        sig = str(c.get("signal") or "topic")
        if sig == "location" or _NOT_A_CRITERION.search(name):
            continue  # a place or a role, already captured as locations / persona.roles
        try:
            weight = max(0.5, min(3.0, float(c.get("weight") or 1)))
        except (TypeError, ValueError):
            weight = 1.0
        terms = [w for w in re.findall(r"[a-z][a-z0-9+-]{2,}", name.lower()) if w not in _STOP][:6]
        crits.append(FitCriterion(name=name, signal=sig if _known_signal(sig) else "topic", weight=weight, terms=terms,
                                  must=bool(c.get("required"))))
    cons = data.get("constraints") if isinstance(data.get("constraints"), dict) else {}
    avoid = [str(a) for a in data.get("anti_patterns") or []] + [str(a) for a in cons.get("avoid") or []]
    return ProspectProfile(criteria=crits[:4],
                           anti_patterns=list(dict.fromkeys(squash(a, 60).lower() for a in avoid if a.strip()))[:8],
                           decision_roles=[str(r) for r in persona.get("roles") or []][:8] or list(pb.default_roles[:8]))


def card(spec: CampaignSpec, ctx: CampaignContext, max_searches: int) -> str:
    pb = get_playbook(spec.pack)
    where = ", ".join(spec.regions + spec.countries) or "anywhere"
    lines = [
        "[bold]Here is what I understood[/bold]",
        f"  {ctx.understanding or spec.objective}",
        "",
        f"  [bold]Kind of outreach:[/bold] {pb.title} — contacting {'people' if spec.target_kind == 'person' else 'organisations'}"
        + (f"  [yellow](not sure: could also be {', '.join(get_playbook(a).title for a in spec.playbook_alternatives)})[/yellow]"
           if spec.playbook_alternatives and (spec.playbook_confidence or 1) < 0.6 else ""),
        f"  [bold]Each email asks for:[/bold] {spec.ask or '(not set)'}"
        + (f" [dim]({spec.ask_detail.type}{'; ' + spec.ask_detail.link if spec.ask_detail.link else ''}"
           f"{'; ' + spec.ask_detail.availability if spec.ask_detail.availability else ''})[/dim]" if spec.ask_detail.type != "other" else ""),
        f"  [bold]Who:[/bold] {spec.persona.description or ', '.join(spec.topics)}",
        f"     roles that qualify: {', '.join(spec.persona.roles[:6]) or 'any'}",
        f"  [bold]Looking for:[/bold] {', '.join(spec.topics)}",
        f"  [bold]Where:[/bold] {where}    [bold]How many:[/bold] {spec.target_count}",
    ]
    if spec.timeline:
        lines.append(f"  [bold]When:[/bold] {spec.timeline}")
    o, k = spec.offer, spec.constraints
    if o.value_prop or o.proof_points:
        lines.append(f"  [bold]Your offer:[/bold] {o.value_prop}" + (f" — proof: {'; '.join(o.proof_points[:3])}" if o.proof_points else ""))
    limits = [x for x in [f"{k.size_min or 0}-{k.size_max or '∞'} employees" if (k.size_min or k.size_max) else "",
                          ", ".join(k.stages), k.seniority, "remote ok" if k.remote_ok else "",
                          f"within {k.radius_km} km" if k.radius_km else "", "must: " + "; ".join(k.must) if k.must else ""] if x]
    if limits:
        lines.append(f"  [bold]Limits:[/bold] {' · '.join(limits)}")
    if spec.language:
        lines.append(f"  [bold]Emails in:[/bold] {spec.language}  [dim](say 'write in English' to change)[/dim]")
    from brain.fit import describe, profile_for

    prof_lines = describe(profile_for(spec, pb))
    if prof_lines:
        lines.append("  [bold]A good target (each candidate is checked against this):[/bold]")
        lines += [f"    • {l}" for l in prof_lines[:8]]
    if spec.exclude_organizations or ctx.exclude_domains or ctx.exclude_emails or ctx.exclude_names:
        n = len(spec.exclude_organizations) + len(ctx.exclude_domains) + len(ctx.exclude_emails) + len(ctx.exclude_names)
        lines.append(f"  [bold]Never contact:[/bold] {n} entries from your instructions and lists")
    lines.append("")
    lines.append("  [bold]Facts the emails may use[/bold] (nothing else will be claimed):")
    facts = ctx.facts[:10]
    lines += [f"    • {squash(f.text, 110)}  [dim]({f.source})[/dim]" for f in facts] or ["    (none yet — emails will be thin)"]
    if ctx.uploads:
        lines.append("  [bold]Your lists:[/bold]")
        lines += [f"    • {u['summary']}" for u in ctx.uploads]
    if ctx.assumptions:
        lines.append("  [bold yellow]Assumptions — correct any that are wrong:[/bold yellow]")
        lines += [f"    • {a}" for a in ctx.assumptions[:8]]
    from brain.explore import choose_tools

    seed, expand = choose_tools(spec, pb, profile_for(spec, pb), has_seeds=bool(ctx.seed_names))
    tools = ", ".join(seed + [f"{t} (from the best finds)" for t in expand])
    lines.append(f"\n  [dim]Plan: discover with {tools}; at most {max_searches} web searches; you review every draft.[/dim]")
    return "\n".join(lines)


def unprovable_notes(spec: CampaignSpec) -> List[str]:
    """Criteria whose kind of proof no installed source gives: said on the card instead of silently assumed."""
    from brain.tools import TOOLS

    out = []
    for c in spec.profile.criteria:
        if c.signal in ("topic", "location"):
            continue
        if not any(c.signal in t.proves for t in TOOLS.values()):
            out.append(f"No source proves '{c.name}' directly; candidates are checked for it from what their pages say")
    return out


def _record_facts(ctx: CampaignContext, data: Dict[str, Any], sender: SenderProfile, sender_source: str) -> None:
    for f in data.get("sender_facts") or []:
        ctx.add_fact(str(f), "you said")
    for h in sender.highlights:
        ctx.add_fact(h, sender_source)


def run_intake(prompt: str, llm: LLM, ui: UI, sender: SenderProfile, *, upload_paths: List[str] = (),
               target_count: Optional[int] = None, locations: Optional[List[str]] = None,
               playbook: Optional[str] = None, interactive: bool = True, max_searches_ceiling: int = 80,
               sender_source: str = "your profile", max_edits: int = 6, store=None) -> IntakeResult:
    calls_before = llm.calls
    ctx = CampaignContext()
    profile = store.user_profile() if store is not None and hasattr(store, "user_profile") else {}
    if profile:  # remembered from earlier campaigns: pre-fill, and say so on the card
        sender.name = sender.name or str(profile.get("name") or "")
        sender.signature = sender.signature or str(profile.get("signature") or "")
        sender.postal_address = sender.postal_address or str(profile.get("postal_address") or "")
        for x in profile.get("exclusions") or []:
            ctx.exclude(str(x))
        ctx.style_examples += [e for e in profile.get("_edited_emails") or [] if e not in ctx.style_examples][:3]
        if profile.get("exclusions") or profile.get("signature"):
            ctx.note("remembered from earlier campaigns: your signature and standing exclusions")
    ui.show("[dim]Reading your request…[/dim]")
    data = understand(prompt, sender, llm)
    if not data:
        raise ValueError("Could not understand the request; please rephrase it.")

    # ── questions ────────────────────────────────────────────────────────────
    answers: List[str] = []
    questions = _merge_questions(data, validate(data, sender))
    if interactive and questions:
        ui.show(f"\n[bold]{squash(str(data.get('understanding') or ''), 500)}[/bold]")
        ui.show("[yellow]A few questions so I target the right people (Enter accepts the default in brackets):[/yellow]")
        for q in questions:
            ans = ui.ask(f"  {q['question']}", default=q["default"])
            if ans and ans != q["default"]:
                answers.append(f"{q['question']} → {ans}")
            elif q["default"]:
                answers.append(f"{q['question']} → (default) {q['default']}")
            if ans and "cv" in q["question"].lower() and ans.lower().endswith((".pdf", ".docx", ".txt")):
                upload_paths = list(upload_paths) + [ans]
            elif ans and q["question"].startswith("Tell me about yourself"):
                ctx.add_fact(ans, "your answer")
                sender.summary = sender.summary or ans

    # ── uploads ──────────────────────────────────────────────────────────────
    pb = get_playbook(playbook or data.get("playbook") or "general")
    paths = [p for p in upload_paths if p]
    if interactive:
        hints = list(dict.fromkeys(pb.upload_prompts + [str(x) for x in data.get("useful_uploads") or []]))[:5]
        ui.show("\n[yellow]Do you have any older lists that could help?[/yellow] For example:")
        for h in hints:
            ui.show(f"  • {h}")
        more = ui.ask("  File paths (CSV, Excel, JSON, TXT, PDF, DOCX), comma-separated — or Enter to skip")
        paths += [p.strip() for p in more.split(",") if p.strip()]
    seeds_contacts: List[Contact] = []
    loaded: List[up.Upload] = []
    for path in paths:
        try:
            u = up.classify(up.load(path))
        except Exception as e:
            ui.show(f"  [red]Could not read {path}: {e}[/red]")
            continue
        if interactive:
            choice = ui.ask(f"  {u.name}: I'll use it as [bold]{u.role}[/bold] ({u.reason}). Enter to accept, or type "
                            f"one of {', '.join(up.ROLES)}", default="").lower()
            if choice in up.ROLES:
                u.role, u.reason = choice, "set by you"
        loaded.append(u)
        seeds_contacts += up.apply(u, ctx)
    about = up.sender_text(loaded)
    if about:
        extra_profile = sender_from_text(about, llm)
        for h in extra_profile.highlights:
            sender.highlights.append(h)
            ctx.add_fact(h, "your uploaded document")
        sender.summary = sender.summary or extra_profile.summary
        sender.name = sender.name or extra_profile.name

    # ── refine ───────────────────────────────────────────────────────────────
    uploads_summary = "\n".join(u["summary"] for u in ctx.uploads)
    if answers or ctx.uploads:
        data = refine(data, prompt, sender, llm, answers=answers, uploads_summary=uploads_summary, seeds=ctx.seed_names)

    cid = new_campaign_id()
    if profile.get("tone") and not data.get("tone"):
        data["tone"] = profile["tone"]
    if profile.get("language") and not data.get("language"):
        data["language"] = profile["language"]
    spec = build(data, sender, campaign_id=cid, target_count=target_count, locations=locations, playbook=playbook)
    ctx.understanding = squash(str(data.get("understanding") or ""), 600)
    ctx.target_brief = spec.persona.description
    ctx.assumptions = [str(a) for a in data.get("assumptions") or []][:10]
    _record_facts(ctx, data, sender, sender_source)
    for org in spec.exclude_organizations:
        ctx.exclude(org)
    ctx.assumptions += unprovable_notes(spec)
    max_searches = search_budget(spec.target_count, max_searches_ceiling)

    # ── verify ───────────────────────────────────────────────────────────────
    if not interactive:
        spec.confirmed = True
        return IntakeResult(spec, ctx, seeds_contacts, only_seeds=False, llm_calls=llm.calls - calls_before)
    for _ in range(max_edits):
        ui.show("\n" + card(spec, ctx, max_searches))
        reply = ui.ask("\n[bold]Start with this? [green]yes[/green] / describe what to change / [red]quit[/red][/bold]",
                       default="yes")
        low = reply.strip().lower()
        if low in ("y", "yes", "ok", "okay", "go", "start", "looks good", "correct"):
            spec.confirmed = True
            break
        if low in ("q", "quit", "exit", "cancel"):
            raise KeyboardInterrupt("cancelled by user")
        data = refine(data, prompt, sender, llm, edits=[reply], uploads_summary=uploads_summary, seeds=ctx.seed_names)
        spec = build(data, sender, campaign_id=cid, target_count=target_count, locations=None if "where" in low or
                     "location" in low else locations, playbook=None if "playbook" in low else playbook)
        ctx.understanding = squash(str(data.get("understanding") or ""), 600)
        ctx.target_brief = spec.persona.description
        ctx.assumptions = [str(a) for a in data.get("assumptions") or []][:10]
        _record_facts(ctx, data, sender, sender_source)
        max_searches = search_budget(spec.target_count, max_searches_ceiling)
    if not spec.confirmed:
        raise KeyboardInterrupt("not confirmed")
    if store is not None and hasattr(store, "save_user_profile"):
        store.save_user_profile({"name": sender.name, "tone": spec.tone, "typical_asks": [spec.ask] if spec.ask else []})
    only = bool(seeds_contacts) and interactive and ui.ask(
        f"\nYou gave me {len(seeds_contacts)} contacts. Work only on your list (skip searching)? yes/no",
        default="no").lower().startswith("y")
    return IntakeResult(spec, ctx, seeds_contacts, only_seeds=only, llm_calls=llm.calls - calls_before)
