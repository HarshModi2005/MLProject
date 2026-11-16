"""
Email drafting: one LLM call per email, checked deterministically.

The model writes only the middle of the email. The salutation and sign-off are added in code, so
they cannot be wrong. Every draft then goes through checks that need no LLM: length, banned filler,
leftover placeholders, and grounding — any quoted title or number must come from the sender's
profile or the contact's sourced evidence. A draft that fails is retried once with the failures
spelled out; if it still fails it is kept but marked as needing attention, never silently
replaced with a generic template.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from brain.llm import LLM, LLMError
from brain.models import CampaignSpec, Contact, Draft
from brain.packs import Pack
from brain.textutil import clean_name, squash, strip_accents

log = logging.getLogger("brain.draft")

BANNED = [
    "i hope this email finds you well", "i hope this message finds you well", "i am writing to express",
    "i came across your profile", "esteemed", "groundbreaking", "cutting-edge", "passionate about",
    "delve", "synergy", "i would be honored", "to whom it may concern",
]

SYSTEM = """You write the body of a short, specific cold email on behalf of the sender.

Rules:
- Use ONLY facts given under SENDER and RECIPIENT. Never invent projects, results, numbers, names or shared history.
- Describe the recipient's work only in terms the evidence supports. Do not attribute topics, goals or findings to them
  that the evidence does not state (e.g. do not say their work is "about privacy" unless the evidence says so).
- Do not claim the sender's experience "aligns with", "matches" or "is directly relevant to" the recipient's work, and
  do not inflate it ("extensive", "deep expertise"); state what the sender did and let the reader judge.
- Each sender fact is labelled with where it happened ("Sifchain Finance: ...", "Project: ..."). Never move a fact to a
  different label: a personal project must not be described as work done at an employer.
- Do not stretch the sender's experience to look relevant. If no sender fact relates to the recipient's work, say so
  plainly and offer the closest real experience as transferable skills — an honest gap reads better than a forced link.
- Refer to the recipient's work using the HOOK. If you name a paper or project title, copy it exactly.
- Structure: (1) why this person, tied to the hook; (2) the one or two sender facts most relevant to THEIR work; (3) the ask, in one sentence, easy to answer.
- Plain text. No greeting line, no sign-off, no bullet points, no placeholders like [Name].
- No filler openers ("I hope this email finds you well"), no flattery adjectives.
- Stay within the word range.

Return JSON: {"subject": "<under 9 words, specific, no clickbait>", "body": "<paragraphs separated by blank lines>"}"""


def salutation(contact: Contact, pack: Pack) -> str:
    if contact.kind == "org":
        return f"Hello {contact.name} team,"
    tokens = clean_name(contact.name).split()
    if not tokens:
        return "Hello,"
    if pack.salutation == "professor":
        return f"Dear Professor {tokens[-1]},"
    return f"Hi {tokens[0]},"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", strip_accents(text).lower())


_EMPLOYER = re.compile(r"\bat\s+((?:[A-Z][\w&.-]*)(?:\s+[A-Z][\w&.-]*){0,3})")
_TERM = re.compile(r"[a-z][a-z0-9+-]{2,}")
_GENERIC = set("""
built build building created create developed develop designed design architected implemented implement led lead
worked work working made using used use enabling enable enabled improving improve improved achieving achieved achieve
system systems component components module modules application applications app apps project projects pipeline
pipelines model models framework frameworks tool tools platform platforms service services solution solutions
data scalable comprehensive robust efficient advanced modern custom multiple various several complex large high
real time end based driven powered integrated integration integrating interface interfaces approach approaches
experience skills team role production performance results accuracy research learning machine deep neural network
networks python and the for with from into that this which their across over under ai ml api apis web code open
source""".split())


_NON_EMPLOYER = {"project", "projects", "personal project", "research", "award", "awards", "education",
                 "achievement", "achievements", "hackathon", "competition", "publication", "skills", "other"}


def _employers(highlights: List[str], summary: str) -> dict:
    """Employer -> indices of the highlights done there. Uses the '<where>: <what>' labels the sender
    parser produces; falls back to 'at <Org>' phrases for profiles written without labels."""
    labelled: dict = {}
    for i, h in enumerate(highlights):
        if ":" in h[:60]:
            where = h.split(":", 1)[0].strip()
            if where and where.lower() not in _NON_EMPLOYER and len(where) > 2:
                labelled.setdefault(where, set()).add(i)
    if labelled:
        return labelled
    found: dict = {}
    for m in _EMPLOYER.finditer(" ".join(highlights) + " " + summary):
        org = m.group(1).strip(" .,")
        if len(org) > 3:
            found[org] = {i for i, h in enumerate(highlights) if org in h}
    return found


def conflations(body: str, highlights: List[str], summary: str = "", recipient_text: str = "") -> List[str]:
    """Sentences that credit one of the sender's employers with work from a different CV item.

    'At Sifchain Finance I built a stock predictor with LSTM and GRU' is flagged when LSTM/GRU/stock appear only in a
    separate project. Deterministic and free; catches what LLM checkers repeatedly miss."""
    employers = _employers(highlights, summary)

    def terms(text: str) -> set:
        return {t for t in _TERM.findall(text.lower()) if t not in _GENERIC}

    recipient_terms = terms(recipient_text)
    summary_sentences = re.split(r"(?<=[.!?])\s+", summary)
    problems = []
    for org, own_idx in employers.items():
        other = [h for i, h in enumerate(highlights) if i not in own_idx]
        if not own_idx or not other:
            continue
        own_text = [highlights[i] for i in own_idx] + [x for x in summary_sentences if org in x]
        own_terms = set().union(*(terms(x) for x in own_text)) | terms(org)
        other_terms = set().union(*(terms(x) for x in other))
        # Check clause by clause: "At X I did A, and in a separate project I did B" is accurate.
        clauses = re.split(r"(?<=[.!?;])\s+|,\s*(?:and|while|but|whereas)\s+|\s+(?:and\s+)?(?:separately|in a separate|"
                           r"outside (?:of )?(?:that|work)|in my own time|as a personal project)\b", body)
        for sentence in clauses:
            if not sentence or org not in sentence:
                continue
            foreign = (terms(sentence) & other_terms) - own_terms - recipient_terms
            if len(foreign) >= 2:
                problems.append(f"credits {org} with work from a different CV item ({', '.join(sorted(foreign)[:4])})")
    return problems


def clean_body(body: str, contact: Contact) -> str:
    """Remove an addressee line the model adds on its own ('Reskilll team,' / 'Dear Priya,') — the code
    writes the greeting — and collapse blank runs."""
    body = re.sub(r"\n{3,}", "\n\n", body.strip())
    first, _, rest = body.partition("\n")
    words = first.strip().rstrip(",").split()
    names = {w.lower() for w in re.findall(r"\w+", contact.name)}
    if first.strip().endswith(",") and len(words) <= 6 and (
            re.match(r"(dear|hi|hello|hey)\b", first.strip(), re.I) or {w.lower().strip(",") for w in words} & (names | {"team"})):
        body = rest.strip()
    return body


def check(body: str, subject: str, contact: Contact, spec: CampaignSpec) -> List[str]:
    """Deterministic quality and grounding checks. Returns a list of problems (empty = pass)."""
    issues: List[str] = []
    words = len(body.split())
    lo, hi = int(spec.word_target * 0.4), int(spec.word_target * 1.45)  # short is fine for a cold email
    if words < lo:
        issues.append(f"too short ({words} words, need at least {lo})")
    if words > hi:
        issues.append(f"too long ({words} words, limit {hi})")
    low = body.lower()
    for phrase in BANNED:
        if phrase in low:
            issues.append(f"contains filler phrase '{phrase}'")
    if re.search(r"\[[^\]]{2,40}\]|\{[^}]{2,40}\}|<[^>]{2,30}>", body + subject):
        issues.append("contains an unfilled placeholder")
    if re.match(r"\s*(dear|hi|hello)\b", low):
        issues.append("body must not start with a greeting")
    if not subject.strip():
        issues.append("missing subject")

    raw_facts = " ".join(
        [spec.sender.facts_text(), spec.objective, spec.ask, spec.timeline or "", contact.name, contact.org or "",
         contact.title or "", contact.bio or "", " ".join(contact.topics)]
        + [f"{ev.text} {ev.detail or ''} {ev.year or ''}" for ev in contact.evidence]
    )
    facts = _norm(raw_facts)
    # Anything in quotes is presented as a real title: it must exist in the evidence.
    for quoted in re.findall(r"[\"“]([^\"”]{12,200})[\"”]", body):
        if _norm(quoted).strip() not in facts:
            issues.append(f"quotes a title not found in the evidence: \"{squash(quoted, 60)}\"")
    issues += conflations(body, spec.sender.highlights, spec.sender.summary, contact.match_text())
    # Numbers are claims (GPA, years, counts): each must trace back to the source facts.
    fact_numbers = set(re.findall(r"\d+(?:\.\d+)?", raw_facts))  # raw text: keeps decimals like 0.91 intact
    for num in set(re.findall(r"\d+(?:\.\d+)?", body)):
        if num not in fact_numbers and len(num) > 1:
            issues.append(f"states a number not found in the source facts: {num}")
    return issues


def with_context(spec: CampaignSpec, ctx=None) -> CampaignSpec:
    """The spec as the writer and checkers should see it: sourced facts from the campaign context (the user's
    answers, uploaded decks, CVs) become sender facts, so they can be used *and* are accepted by the checks."""
    if ctx is None or not getattr(ctx, "facts", None):
        return spec
    s2 = spec.model_copy(deep=True)
    have = {h.lower() for h in s2.sender.highlights}
    for f in ctx.facts:
        if f.text.lower() not in have and len(s2.sender.highlights) < 16:
            s2.sender.highlights.append(f.text)
            have.add(f.text.lower())
    return s2


def relevant_highlights(contact: Contact, spec: CampaignSpec, k: int = 2) -> List[str]:
    """Sender facts ordered by overlap with this recipient's evidence (BM25, no LLM)."""
    from brain.rank import bm25_scores

    highlights = [h for h in spec.sender.highlights if h.strip()]
    if not highlights:
        return []
    recipient = contact.match_text()
    scores = bm25_scores(recipient, highlights)
    ranked = sorted(zip(scores, highlights), key=lambda x: x[0], reverse=True)
    return [h for score, h in ranked[:k] if score > 0] or [ranked[0][1]]


def _prompt(contact: Contact, spec: CampaignSpec, pack: Pack, style=()) -> str:
    s = spec.sender
    top_facts = relevant_highlights(contact, spec)
    evidence = "\n".join(
        f"- {ev.text}" + (f" ({ev.year})" if ev.year else "") + (f" — {squash(ev.detail, 220)}" if ev.detail else "")
        for ev in contact.evidence[:4]
    )
    lo, hi = int(spec.word_target * 0.8), int(spec.word_target * 1.2)
    return f"""SENDER
Name: {s.name}
Who: {s.headline}
Background: {squash(s.summary, 700)}
Closest to this recipient's work (pre-ranked; prefer these):
{chr(10).join('- ' + squash(h, 260) for h in top_facts) or '- (none)'}
Other highlights (use only if clearly more relevant):
{chr(10).join('- ' + squash(h, 200) for h in s.highlights if h not in top_facts) or '- (none)'}
Skills: {', '.join(s.skills[:10])}

GOAL
Objective: {spec.objective}
Ask: {spec.ask or spec.objective}
Timing: {spec.timeline or 'not specified'}
Tone: {spec.tone}

RECIPIENT ({'an organisation, written to through its shared inbox ' + (best.address if (best := contact.best_email) else '') + ' — address the team, not a named person' if contact.kind == 'org' else 'a person'})
Name: {contact.name}
Role: {contact.title or ('organisation' if contact.kind == 'org' else 'unknown')} at {contact.org or 'unknown'}
Works on: {', '.join(contact.topics[:6]) or 'see evidence'}
HOOK (most relevant fact): {contact.hook or (contact.evidence[0].text if contact.evidence else 'none — keep the reference to their work general')}
Evidence:
{evidence or '- (none)'}
Why they fit: {'; '.join(contact.reasons) or 'n/a'}

STYLE NOTES: {pack.writing_notes}
LENGTH: {lo}-{hi} words for the body.""" + (
        "\n\nAN EMAIL THE SENDER WROTE BEFORE (match its tone, directness and length; do not copy its content):\n"
        + squash(style[0], 900) if style else "")


def draft_email(contact: Contact, spec: CampaignSpec, pack: Pack, llm: LLM, ctx=None) -> Optional[Draft]:
    """Returns a Draft (possibly with issues), or None if the LLM could not be reached at all."""
    best = contact.best_email
    spec = with_context(spec, ctx)
    style = list(getattr(ctx, "style_examples", []) or [])
    prompt = _prompt(contact, spec, pack, style)
    issues: List[str] = []
    subject = body = ""
    for attempt in range(2):
        user = prompt if attempt == 0 else (
            prompt + "\n\nYour previous draft was rejected for these reasons — fix all of them:\n- "
            + "\n- ".join(issues) + f"\n\nPrevious draft:\n{body}"
        )
        try:
            data = llm.json(system=SYSTEM, user=user, purpose="draft", tier="strong", temperature=0.5,
                            max_tokens=700)
        except LLMError as e:
            log.warning("draft failed for %s: %s", contact.name, e)
            return None
        subject = squash(str(data.get("subject") or ""), 120)
        body = clean_body(str(data.get("body") or ""), contact)
        issues = check(body, subject, contact, spec)
        if not issues:
            break
    full = f"{salutation(contact, pack)}\n\n{body}\n\nBest regards,\n{spec.sender.name}".strip()
    return Draft(contact_id=contact.id, to=best.address if best else "", subject=subject, body=full, issues=issues)


# ── fact-check pass (batched) ────────────────────────────────────────────────

VERIFY_SYSTEM = """You fact-check cold emails before they are sent.
For each draft, list every claim that is NOT supported by the facts provided for it:
 - statements about the recipient's work, interests or findings that their EVIDENCE does not state;
 - statements about the sender's experience, skills or results that SENDER FACTS do not state
   (an exaggeration such as "extensive experience" from a single project counts);
 - invented connections presented as fact ("which uses the same technique as my project") when the evidence does not show it.
 - conflations: attributing one sender project's methods or results to a different job or project
   (e.g. saying work was done "at Company X" when SENDER FACTS list it as a separate personal project).
Opinions and intentions ("I am interested in", "I would like to") are fine. Paraphrase is fine if the meaning is preserved.
Return JSON: {"results": [{"id": "<id>", "unsupported": ["<short quote of the unsupported claim>"]}]}
Use an empty list when a draft is fully supported. Include every id exactly once."""


def _recipient_facts(c: Contact) -> str:
    lines = [f"{c.name}; {c.title or ''} {c.org or ''}".strip()]
    if c.topics:
        lines.append("Listed topics: " + ", ".join(c.topics[:6]))
    if c.bio:
        lines.append("Bio: " + squash(c.bio, 300))
    for ev in c.evidence[:4]:
        lines.append(f"- {ev.text}" + (f" ({ev.year})" if ev.year else "") + (f" — {squash(ev.detail, 220)}" if ev.detail else ""))
    return "\n".join(lines)


def find_unsupported(items, spec: CampaignSpec, llm: LLM, batch_size: int = 4) -> dict:
    """items: list of (contact, draft). Returns {contact_id: [unsupported claims]} for flagged drafts only.
    Raises LLMError if the checker cannot be reached (callers decide how to treat that)."""
    sender_facts = spec.sender.facts_text()[:3000]
    flagged: dict = {}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        blocks = []
        for c, d in batch:
            body = d.body.split("\n", 1)[1] if "\n" in d.body else d.body  # drop the salutation
            blocks.append(f"### id: {c.id}\nEVIDENCE ABOUT RECIPIENT:\n{_recipient_facts(c)}\nDRAFT:\n{body}")
        user = f"SENDER FACTS:\n{sender_facts}\nGoal: {spec.objective}; timing: {spec.timeline or 'n/a'}\n\n" + "\n\n".join(blocks)
        # Careful reading matters more here than anywhere else, so the checker thinks harder.
        data = llm.json(system=VERIFY_SYSTEM, user=user, purpose="verify", tier="strong", reasoning="medium",
                        max_tokens=250 * len(batch) + 1200)
        ids = {c.id for c, _ in batch}
        for item in (data.get("results") or []) if isinstance(data, dict) else []:
            if not isinstance(item, dict) or str(item.get("id")) not in ids:
                continue
            claims = [squash(str(x), 160) for x in (item.get("unsupported") or []) if str(x).strip()]
            if claims:
                flagged[str(item["id"])] = claims[:5]
    return flagged


def verify_and_fix(contacts, drafts: dict, spec: CampaignSpec, pack: Pack, llm: LLM, ctx=None) -> dict:
    """Fact-check every passing draft; rewrite flagged ones once; re-check the rewrites.
    Mutates drafts in place. Returns stats."""
    stats = {"checked": 0, "flagged": 0, "fixed": 0, "still_flagged": 0}
    spec = with_context(spec, ctx)
    by_id = {c.id: c for c in contacts}
    items = [(by_id[cid], d) for cid, d in drafts.items() if d.ok and cid in by_id]
    if not items:
        return stats
    try:
        flagged = find_unsupported(items, spec, llm)
    except LLMError as e:
        log.warning("fact-check unavailable: %s", e)
        for _, d in items:
            d.issues.append("not fact-checked (checker unavailable)")
        return stats
    stats["checked"], stats["flagged"] = len(items), len(flagged)
    rewritten = []
    for cid, claims in flagged.items():
        c, d = by_id[cid], drafts[cid]
        new = _rewrite(c, d, claims, spec, pack, llm)
        if new is None:
            d.issues += [f"unsupported claim: {x}" for x in claims]
            continue
        drafts[cid] = new
        if new.ok:
            rewritten.append((c, new))
    if rewritten:
        try:
            again = find_unsupported(rewritten, spec, llm)
        except LLMError:
            again = {c.id: ["could not re-check after rewrite"] for c, _ in rewritten}
        for c, d in rewritten:
            if c.id in again:
                d.issues += [f"unsupported claim: {x}" for x in again[c.id]]
                stats["still_flagged"] += 1
            else:
                stats["fixed"] += 1
    return stats


def _rewrite(c: Contact, d: Draft, claims, spec: CampaignSpec, pack: Pack, llm: LLM):
    body = d.body.split("\n\n", 1)[1].rsplit("\n\nBest regards", 1)[0] if "\n\n" in d.body else d.body
    user = (_prompt(c, spec, pack) + "\n\nA fact-checker found these claims unsupported by the facts above:\n- "
            + "\n- ".join(claims) + f"\n\nRewrite the draft without them (do not replace them with other unsupported claims).\n\nDraft:\n{body}")
    try:
        data = llm.json(system=SYSTEM, user=user, purpose="draft_fix", tier="strong", temperature=0.3, max_tokens=700)
    except LLMError as e:
        log.warning("rewrite failed for %s: %s", c.name, e)
        return None
    subject = squash(str(data.get("subject") or d.subject), 120)
    new_body = clean_body(str(data.get("body") or ""), c)
    issues = check(new_body, subject, c, spec)
    full = f"{salutation(c, pack)}\n\n{new_body}\n\nBest regards,\n{spec.sender.name}".strip()
    return Draft(contact_id=c.id, to=d.to, subject=subject, body=full, issues=issues)
