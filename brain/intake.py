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
from brain.models import CampaignSpec, Contact, FitCriterion, ProspectProfile, SenderProfile, TargetPersona
from brain.playbooks import PLAYBOOKS, catalog_text, get_playbook
from brain.searchplan import search_budget
from brain.spec import new_campaign_id, sender_from_text
from brain.textutil import split_locations, squash

log = logging.getLogger("brain.intake")

_SCHEMA = """{
 "playbook": "<one name from PLAYBOOKS>",
 "understanding": "<2-3 plain sentences restating what the user wants, to whom, and why — as you would read it back to them>",
 "objective": "<the outcome the user wants, one sentence>",
 "ask": "<the concrete request each email makes, one sentence>",
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
   "signal": "past_sponsorship|hiring|portfolio_fit|recent_coverage|publications|audience|spoke_before|local_business|funding_focus|open_source|topic",
   "weight": <1-3>, "required": <true only if the user said it is a must>}],
 "anti_patterns": ["<kinds of organisation/person that look relevant but are poor targets, e.g. 'event agency', 'job board'>"],
 "event_queries": ["<only when targets are sponsors or speakers: 2-4 queries for SIMILAR PAST EVENTS' sponsor/speaker pages>"],
 "preferred_inboxes": ["<if targets are organisations: shared inbox names that suit this ask, best first>"],
 "sender_facts": ["<facts about the sender or the offer that the USER stated (numbers, achievements, what is offered), one per item, in their words>"],
 "assumptions": ["<everything you inferred that the user did NOT say: the audience, location, roles, counts...>"],
 "questions": [{"question": "<short question>", "why": "<what changes with the answer>", "default": "<what you will assume if they skip>"}]
}"""

UNDERSTAND_SYSTEM = f"""You configure a cold-outreach campaign from a user's request. Read it carefully and precisely.

PLAYBOOKS:
{{catalog}}

Return JSON with exactly these keys:
{_SCHEMA}

Rules:
- Never invent facts about the sender. sender_facts may only contain what the user or their profile states.
- Separate what was said from what you assumed: anything not stated goes into "assumptions".
- Ask at most 4 questions, only ones whose answers would change who is contacted or what the email says.
  Do not ask about things already stated. Give every question a sensible default.
- Topics describe the TARGETS (e.g. "dental clinics", "edtech startups"), not the sender's product."""

REFINE_SYSTEM = f"""You update a cold-outreach campaign configuration with new information from the user (answers,
corrections, uploaded lists). Apply corrections exactly; when the user answers a question, remove the corresponding
assumption and question. Keep everything else unchanged unless the new information contradicts it.

PLAYBOOKS:
{{catalog}}

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


def understand(prompt: str, sender: SenderProfile, llm: LLM) -> Dict[str, Any]:
    data = llm.json(system=UNDERSTAND_SYSTEM.replace("{catalog}", catalog_text()), user=_user_block(prompt, sender),
                    purpose="intake", tier="strong", max_tokens=2200)
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
    out = llm.json(system=REFINE_SYSTEM.replace("{catalog}", catalog_text()), user=_user_block(prompt, sender, extra),
                   purpose="intake_refine", tier="strong", max_tokens=2200)
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
    return qs


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
    countries, regions = split_locations(locations if locations else [str(x) for x in data.get("locations") or []])
    persona = data.get("persona") or {}
    count = target_count or data.get("target_count")
    try:
        count = max(1, min(200, int(count))) if count else 25
    except (TypeError, ValueError):
        count = 25
    kind = data.get("target_kind") if data.get("target_kind") in ("person", "org") else pb.target_kind
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
    )


_SIGNALS = {"past_sponsorship", "hiring", "portfolio_fit", "recent_coverage", "publications", "audience", "spoke_before",
            "local_business", "funding_focus", "open_source", "topic"}
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
        try:
            weight = max(0.5, min(3.0, float(c.get("weight") or 1)))
        except (TypeError, ValueError):
            weight = 1.0
        terms = [w for w in re.findall(r"[a-z][a-z0-9+-]{2,}", name.lower()) if w not in _STOP][:6]
        crits.append(FitCriterion(name=name, signal=sig if sig in _SIGNALS else "topic", weight=weight, terms=terms,
                                  must=bool(c.get("required"))))
    return ProspectProfile(criteria=crits[:4],
                           anti_patterns=[squash(str(a), 60).lower() for a in data.get("anti_patterns") or []][:6],
                           decision_roles=[str(r) for r in persona.get("roles") or []][:8] or list(pb.default_roles[:8]))


def card(spec: CampaignSpec, ctx: CampaignContext, max_searches: int) -> str:
    pb = get_playbook(spec.pack)
    where = ", ".join(spec.regions + spec.countries) or "anywhere"
    lines = [
        "[bold]Here is what I understood[/bold]",
        f"  {ctx.understanding or spec.objective}",
        "",
        f"  [bold]Kind of outreach:[/bold] {pb.title} — contacting {'people' if spec.target_kind == 'person' else 'organisations'}",
        f"  [bold]Each email asks for:[/bold] {spec.ask or '(not set)'}",
        f"  [bold]Who:[/bold] {spec.persona.description or ', '.join(spec.topics)}",
        f"     roles that qualify: {', '.join(spec.persona.roles[:6]) or 'any'}",
        f"  [bold]Looking for:[/bold] {', '.join(spec.topics)}",
        f"  [bold]Where:[/bold] {where}    [bold]How many:[/bold] {spec.target_count}",
    ]
    if spec.timeline:
        lines.append(f"  [bold]When:[/bold] {spec.timeline}")
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


def _record_facts(ctx: CampaignContext, data: Dict[str, Any], sender: SenderProfile, sender_source: str) -> None:
    for f in data.get("sender_facts") or []:
        ctx.add_fact(str(f), "you said")
    for h in sender.highlights:
        ctx.add_fact(h, sender_source)


def run_intake(prompt: str, llm: LLM, ui: UI, sender: SenderProfile, *, upload_paths: List[str] = (),
               target_count: Optional[int] = None, locations: Optional[List[str]] = None,
               playbook: Optional[str] = None, interactive: bool = True, max_searches_ceiling: int = 80,
               sender_source: str = "your profile", max_edits: int = 6) -> IntakeResult:
    calls_before = llm.calls
    ctx = CampaignContext()
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
    spec = build(data, sender, campaign_id=cid, target_count=target_count, locations=locations, playbook=playbook)
    ctx.understanding = squash(str(data.get("understanding") or ""), 600)
    ctx.target_brief = spec.persona.description
    ctx.assumptions = [str(a) for a in data.get("assumptions") or []][:10]
    _record_facts(ctx, data, sender, sender_source)
    for org in spec.exclude_organizations:
        ctx.exclude(org)
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
    only = bool(seeds_contacts) and interactive and ui.ask(
        f"\nYou gave me {len(seeds_contacts)} contacts. Work only on your list (skip searching)? yes/no",
        default="no").lower().startswith("y")
    return IntakeResult(spec, ctx, seeds_contacts, only_seeds=only, llm_calls=llm.calls - calls_before)
