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

from brain.safety import guard
from brain.llm import LLM, LLMError
from brain.models import CampaignSpec, Contact, Draft
from brain.packs import Pack
from brain.emailfinder import is_role_address
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
- Structure: (1) why this person, tied to the hook (and WHY NOW if given); (2) the one or two sender facts most relevant
  to THEIR work; (3) exactly ONE ask, in one sentence, easy to answer. If a link or availability is given, use it.
- Mutual connections: mention one ONLY as the evidence states it ("we are both connected to X", "I know X at your
  company"). Never claim someone introduced, referred or recommended the sender unless the evidence says so.
- Never mention the recipient's email address, "inbox", "mailbox" or hope that the message "reaches" anyone. The reader
  must not be able to tell how the address was found or that it is shared.
- Follow the SITUATION rules.
- Plain text. No greeting line, no sign-off, no bullet points, no placeholders like [Name], at most one link.
- No filler openers ("I hope this email finds you well"), no flattery adjectives, no hype, no exclamation marks.
- Short sentences (under 25 words). Stay within the word range. Write in the LANGUAGE given.

Return JSON: {"subjects": ["<3 different subject lines, each under 9 words, specific to this recipient, no clickbait>"],
"body": "<paragraphs separated by blank lines>"}"""


def _greet_name(tokens: List[str]) -> str:
    """'C Jagadeesh' -> 'C Jagadeesh' (an initial is not a first name); 'Priya Nair' -> 'Priya'."""
    if len(tokens) > 1 and len(tokens[0].strip(".")) <= 1:
        return " ".join(tokens)
    return tokens[0]


_NAME_PARTICLES = {"van", "von", "der", "den", "de", "del", "della", "di", "da", "dos", "das", "du", "la", "le",
                   "bin", "ibn", "al", "el", "ter", "ten", "st", "mac"}
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}


def surname(tokens: List[str]) -> str:
    """'Anna van der Berg' -> 'van der Berg'; 'Rao' -> 'Rao'; 'John Smith Jr.' -> 'Smith'."""
    toks = [t for t in tokens]
    while len(toks) > 1 and toks[-1].strip(".,").lower() in _NAME_SUFFIXES:
        toks.pop()
    i = len(toks) - 1
    while i > 1 and toks[i - 1].lower() in _NAME_PARTICLES:  # i > 1: keep a first name in front
        i -= 1
    return " ".join(toks[i:])


def team_name(name: str) -> str:
    """'The TWIML AI Podcast (formerly This Week in ...)' -> 'TWIML AI Podcast'."""
    n = re.sub(r"\s*[\(\[].*?[\)\]]", "", name or "")
    n = re.split(r"\s+[|–—-]\s+|:\s", n)[0]
    n = re.sub(r"^the\s+", "", n.strip(), flags=re.I)
    return n.strip() or name


DM_TRUST = 0.7  # below this a found "decision maker" is not greeted by name (search snippets can name a stranger)


def trusted_dm(contact: Contact) -> dict:
    """The decision maker a draft may address by name: found by a paid provider, or by a search that named the
    organisation and the role together. Records from before the confidence was kept count as unverified."""
    dm = contact.decision_maker or {}
    if not dm.get("name"):
        return {}
    conf = dm.get("confidence")
    if conf is None:
        conf = 1.0 if dm.get("source") == "hunter" else 0.5
    return dm if float(conf) >= DM_TRUST else {}


def salutation(contact: Contact, pack: Pack) -> str:
    from brain.textutil import looks_like_person_name

    if contact.kind == "org":
        dm = trusted_dm(contact)
        dm_name = clean_name(dm.get("name") or "")
        first = dm_name.split() if looks_like_person_name(dm_name) else []  # "At SIM" is not someone to greet
        if first:  # found the right person but only the shared inbox: address them through it
            return f"Hi {_greet_name(first)},"
        best = contact.best_email
        shared = bool(best) and (best.method == "inbox" or is_role_address(best.address))
        if looks_like_person_name(contact.name) and not shared:  # a one-person business listed under its owner's name
            return f"Hi {_greet_name(clean_name(contact.name).split())},"  # (never through info@: "Hi OM," for "Om Happy Teeth")
        return f"Hello {team_name(contact.name)} team,"
    tokens = clean_name(contact.name).split()
    if not tokens:
        return "Hello,"
    if pack.salutation == "professor":
        title = (contact.title or "").lower()
        if re.search(r"professor|\bprof\b|lecturer|\bfaculty\b|\bchair\b", title):
            return f"Dear Professor {surname(tokens)},"
        if re.search(r"\bdr\b|\bphd\b|doctor|researcher|scientist|fellow|investigator", title):
            return f"Dear Dr. {surname(tokens)},"
        return f"Dear {' '.join(tokens)},"  # no title in the evidence: do not promote or demote anyone
    return f"Hi {_greet_name(tokens)},"


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


_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_META = re.compile(r"\b(?:shared|team|general|generic|your|this|the)?\s*(?:e-?mail\s+)?(?:inbox|mailbox)\b|"
                   r"\b(?:hoping|hope|trust)\s+(?:this|that|my)\s+(?:message|email|note)?\s*(?:finds|reaches|gets to|"
                   r"makes it to|lands with)\b|\breaches?\s+(?:you|your team|the right (?:person|team))\b", re.I)


def leaks(text: str) -> List[str]:
    """How the email was addressed must not show in it: a raw address, or talk of an inbox or of hoping it arrives."""
    out: List[str] = []
    if _ADDRESS.search(text):
        out.append("shows an email address in the text")
    if _META.search(text):
        out.append("talks about the inbox or about the message reaching someone")
    return out


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
    english = not spec.language or spec.language.lower() == "english"
    for phrase in BANNED if english else []:
        if phrase in low:
            issues.append(f"contains filler phrase '{phrase}'")
    if re.search(r"\[[^\]]{2,40}\]|\{[^}]{2,40}\}|<[^>]{2,30}>", body + subject):
        issues.append("contains an unfilled placeholder")
    if re.match(r"\s*(dear|hi|hello)\b", low):
        issues.append("body must not start with a greeting")
    if not subject.strip():
        issues.append("missing subject")
    issues += leaks(body + "\n" + subject)

    o, a = spec.offer, spec.ask_detail
    raw_facts = " ".join(
        [spec.sender.facts_text(), spec.objective, spec.ask, spec.timeline or "", o.value_prop, " ".join(o.proof_points),
         o.terms, o.audience, a.link or "", a.availability or "", a.deadline or "", contact.name, contact.org or "",
         contact.title or "", contact.bio or "", " ".join(contact.topics)]
        + [f"{ev.text} {ev.detail or ''} {ev.year or ''}" for ev in contact.evidence]
    )
    facts = _norm(raw_facts)
    # Anything in quotes is presented as a real title: it must exist in the evidence.
    for quoted in re.findall(r"[\"“]([^\"”]{12,200})[\"”]", body):
        if _norm(quoted).strip() not in facts:
            issues.append(f"quotes a title not found in the evidence: \"{squash(quoted, 60)}\"")
    issues += conflations(body, spec.sender.highlights, spec.sender.summary, contact.match_text())
    issues += quality_problems(body, contact, spec)
    # Numbers are claims (GPA, years, counts): each must trace back to the source facts.
    fact_numbers = set(re.findall(r"\d+(?:\.\d+)?", raw_facts))  # raw text: keeps decimals like 0.91 intact
    for num in set(re.findall(r"\d+(?:\.\d+)?", body)):
        if num not in fact_numbers and len(num) > 1:
            issues.append(f"states a number not found in the source facts: {num}")
    return issues


# A second request tacked on after the ask ("Also, if you could connect me with ...").
_SECOND_ASK = re.compile(r"(^|[.?!]\s+)(also|alternatively|additionally|separately|in addition),?\s+"
                         r"(if you could|could you|would you|can you|please|i would (also )?(appreciate|love)|i'?d (also )?(appreciate|love))",
                         re.I)
_SHORTENERS = ("bit.ly", "tinyurl", "t.co/", "goo.gl", "ow.ly", "rb.gy", "cutt.ly", "is.gd")
_WARM_OVERCLAIM = re.compile(r"\b(introduced me|suggested (that )?I (reach|contact|write)|referred me|recommended (that )?I|"
                             r"asked me to (reach|contact|write)|told me to (reach|contact))", re.I)
_STOPWORDS = {"this", "that", "with", "from", "your", "their", "have", "about", "would", "could", "which", "there", "where",
              "what", "when", "will", "into", "they", "them", "were", "been", "more", "also", "such", "only"}


def _recipient_terms(contact: Contact) -> set:
    """Content words specific to this recipient (from the hook and evidence), not their name or generic words."""
    own = {w.lower() for w in re.findall(r"[A-Za-z]+", contact.name)}
    text = " ".join([contact.hook or ""] + [f"{e.text} {e.detail or ''}" for e in contact.evidence[:6]] + contact.topics)
    return {t for t in _TERM.findall(text.lower()) if len(t) > 3 and t not in _GENERIC and t not in _STOPWORDS
            and t not in own}


def quality_problems(body: str, contact: Contact, spec: CampaignSpec) -> List[str]:
    """Personalisation, one ask, deliverability and readability — deterministic, free."""
    out: List[str] = []
    english = not spec.language or spec.language.lower() == "english"
    terms = _recipient_terms(contact)
    if terms and english and not ({t for t in _TERM.findall(body.lower())} & terms):
        out.append("not personalised: mentions nothing specific from the recipient's evidence")
    if body.count("?") > 2 or _SECOND_ASK.search(body):
        out.append("more than one ask")
    links = re.findall(r"https?://\S+|www\.\S+", body)
    if len(links) > 1:
        out.append(f"{len(links)} links (at most one)")
    if any(s in body.lower() for s in _SHORTENERS):
        out.append("uses a link shortener (spam filters penalise them)")
    if body.count("!") > 1:
        out.append("more than one exclamation mark")
    if any(w.isupper() and len(w) > 5 and w.isalpha() for w in body.split()):
        out.append("ALL-CAPS word")
    if english:
        low = body.lower()
        hit = next((w for w in SPAM_WORDS if w in low), None)
        if hit:
            out.append(f"spam-trigger phrase '{hit}'")
        sentences = [x for x in re.split(r"(?<=[.!?])\s+", body.strip()) if x.strip()]
        if sentences and sum(len(x.split()) for x in sentences) / len(sentences) > 28:
            out.append("sentences too long on average (over 28 words)")
        for para in [p for p in body.split("\n\n") if p.strip()]:
            if len([x for x in re.split(r"(?<=[.!?])\s+", para.strip()) if x]) > 5:
                out.append("a paragraph has more than 5 sentences")
                break
    evidence = " ".join(e.text for e in contact.evidence).lower()
    if _WARM_OVERCLAIM.search(body) and not re.search(r"introduc|referr|recommend", evidence):
        out.append("claims an introduction or referral the evidence does not show")
    return out


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
    """Sender facts for this recipient: the one the judge chose (it read both sides) first, then BM25 overlap."""
    from brain.rank import bm25_scores

    highlights = [h for h in spec.sender.highlights if h.strip()]
    if not highlights:
        return []
    chosen = []
    if contact.sender_ref is not None and 0 <= contact.sender_ref < len(spec.sender.highlights):
        h = spec.sender.highlights[contact.sender_ref]
        if h.strip():
            chosen.append(h)
    recipient = contact.match_text()
    scores = bm25_scores(recipient, highlights)
    ranked = sorted(zip(scores, highlights), key=lambda x: x[0], reverse=True)
    rest = [h for score, h in ranked if score > 0 and h not in chosen]
    out = (chosen + rest)[:k]
    return out or [ranked[0][1]]


# ── situations: who exactly is reading, and how they were reached ────────────

GUESSED_METHODS = ("pattern", "pattern_confirmed", "pattern_global", "pattern_prior")

SITUATION_RULES = {
    "person_own_address": "Writing to the person's own address: be direct and personal.",
    "person_guessed_address": ("The address follows the organisation's usual format and may not reach them: do not imply "
                               "you know them; if it is not their area, ask them to point you to the right person."),
    "decision_maker_via_inbox": ("The email goes to the organisation's shared team inbox but is addressed to the named "
                                 "person: do not comment on the address. Keep ONE ask; you may end it with a short clause "
                                 "such as 'or point me to whoever handles this' — never a second request."),
    "org_inbox": ("The email goes to a shared team inbox: write to the team and make it easy to forward. The ONE ask is "
                  "the main ask, in full. You may end with a short clause such as the SHARED-INBOX ASK (or point me to "
                  "whoever handles this) — never a second request, and never replace the main ask with a request for "
                  "a referral."),
    "warm_direct": "The sender is already connected to this person (see evidence): mention it once, plainly, first.",
    "warm_intro": ("The sender knows someone at this organisation (see evidence): you may mention that connection once, "
                   "plainly; do not claim they referred or introduced the sender."),
}


def situation(contact: Contact) -> str:
    warm = contact.signals.get("warm_path", 0)
    if warm >= 1.0:
        return "warm_direct"
    if warm > 0:
        return "warm_intro"
    best = contact.best_email
    if contact.kind == "org":
        shared = bool(best) and (best.method == "inbox" or is_role_address(best.address))
        if not shared:
            return "person_own_address"  # e.g. a small clinic's own Gmail: write as to its owner
        return "decision_maker_via_inbox" if trusted_dm(contact) else "org_inbox"
    if best and best.method in GUESSED_METHODS:
        return "person_guessed_address"
    return "person_own_address"


def strongest_criterion(contact: Contact) -> Optional[str]:
    if not contact.criteria:
        return None
    name, v = max(contact.criteria.items(), key=lambda kv: kv[1] or 0)
    return f"{name} (support {v:.2f})" if v and v >= 0.5 else None


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
Ask: {spec.ask or spec.objective}{(' [' + a.type + ']') if (a := spec.ask_detail).type != 'other' else ''}
{('Link to include: ' + a.link) if a.link else ''}{(chr(10) + 'Sender availability: ' + a.availability) if a.availability else ''}{(chr(10) + 'Deadline: ' + a.deadline) if a.deadline else ''}
SHARED-INBOX ASK: {(a.inbox_text or 'n/a') if situation(contact) in ('org_inbox', 'decision_maker_via_inbox') else 'n/a (not writing to a team inbox)'}
Timing: {spec.timeline or 'not specified'}
Tone: {spec.tone}
LANGUAGE: {spec.language or 'English'}
{('OFFER: ' + spec.offer.value_prop + ((' | proof: ' + '; '.join(spec.offer.proof_points[:3])) if spec.offer.proof_points else '') + ((' | terms: ' + spec.offer.terms) if spec.offer.terms else '')) if spec.offer.value_prop else ''}

RECIPIENT ({'an organisation, reached through its shared team mailbox; never mention the mailbox or its address' + (' — address the team, not a named person' if not trusted_dm(contact) else '') if contact.kind == 'org' else 'a person'})
Name: {contact.name}
Role: {contact.title or ('organisation' if contact.kind == 'org' else 'unknown')} at {contact.org or 'unknown'}
Works on: {', '.join(contact.topics[:6]) or 'see evidence'}
HOOK (most relevant fact): {contact.hook or (contact.evidence[0].text if contact.evidence else 'none — keep the reference to their work general')}
WHY NOW: {contact.why_now or 'none given — do not invent urgency'}
STRONGEST FIT: {strongest_criterion(contact) or 'n/a'}
Evidence:
{evidence or '- (none)'}
Why they fit: {'; '.join(contact.reasons) or 'n/a'}

SITUATION: {SITUATION_RULES[situation(contact)]}
STYLE NOTES: {pack.writing_notes}
LENGTH: {lo}-{hi} words for the body.""" + (
        "\n\nAN EMAIL THE SENDER WROTE BEFORE (match its tone, directness and length; do not copy its content):\n"
        + squash(style[0], 900) if style else "")


def draft_email(contact: Contact, spec: CampaignSpec, pack: Pack, llm: LLM, ctx=None) -> Optional[Draft]:
    """Returns a Draft (possibly with issues), or None if the LLM could not be reached at all."""
    from brain.safety import scrub

    scrub([contact])
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
            data = llm.json(system=guard(SYSTEM), user=user, purpose="draft", tier="strong", temperature=0.5,
                            max_tokens=700)
        except LLMError as e:
            log.warning("draft failed for %s: %s", contact.name, e)
            return None
        body = clean_body(str(data.get("body") or ""), contact)
        subject = choose_subject(_subjects(data), contact, spec)
        issues = check(body, subject, contact, spec)
        if not issues:
            break
    if "injection_suspect" in contact.flags:
        issues = issues + ["the recipient's source pages contained text trying to instruct an AI; read the sources before sending"]
    if getattr(pack, "commercial", False) and not spec.sender.postal_address.strip():
        # commercial email law (CAN-SPAM and similar) expects a physical address in the footer
        issues = issues + ["commercial email without a sender postal address (add one in the sender profile)"]
    return Draft(contact_id=contact.id, to=best.address if best else "", subject=subject,
                 body=assemble(body, contact, spec, pack), issues=issues)


def _subjects(data) -> List[str]:
    subs = data.get("subjects") if isinstance(data, dict) else None
    out = [squash(str(x), 120) for x in subs] if isinstance(subs, list) else []
    if isinstance(data, dict) and data.get("subject"):
        out.append(squash(str(data["subject"]), 120))
    return [x for x in out if x.strip()]


SPAM_WORDS = ["act now", "limited time", "100% free", "risk-free", "risk free", "guaranteed", "guarantee", "click here",
              "buy now", "winner", "urgent", "no obligation", "earn money", "cash bonus", "special promotion",
              "exclusive deal", "once in a lifetime", "!!!", "$$$", "dear friend", "congratulations"]


def subject_problems(subject: str, contact: Contact, spec: CampaignSpec) -> List[str]:
    out = []
    words = subject.split()
    if not words:
        return ["missing subject"]
    if len(words) > 9:
        out.append("subject longer than 9 words")
    if re.match(r"\s*(re|fwd?)\s*:", subject, re.I):
        out.append("subject pretends to be a reply or forward")
    if "!" in subject or any(w.isupper() and len(w) > 4 for w in words):
        out.append("subject shouts (! or ALL CAPS)")
    low = subject.lower()
    if any(w in low for w in SPAM_WORDS):
        out.append("subject contains a spam-trigger phrase")
    return out


def choose_subject(subjects: List[str], contact: Contact, spec: CampaignSpec) -> str:
    """The first subject that passes the checks, preferring one that names something specific to the recipient."""
    if not subjects:
        return ""
    specific = _recipient_terms(contact) | {w.lower() for w in re.findall(r"[A-Za-z]{4,}", contact.org or "")}
    clean = [x for x in subjects if not subject_problems(x, contact, spec)]
    pool = clean or subjects
    ranked = sorted(pool, key=lambda x: -len({t for t in _TERM.findall(x.lower())} & specific))
    return ranked[0]


_OPT_OUT = {"": "If this isn't relevant, just reply 'no' and I won't write again.",
            "German": "Falls das für Sie nicht relevant ist, antworten Sie einfach mit „nein“ und ich schreibe nicht wieder.",
            "French": "Si ce n'est pas pertinent, répondez simplement « non » et je ne vous écrirai plus.",
            "Spanish": "Si no le resulta relevante, responda «no» y no volveré a escribirle.",
            "Portuguese": "Se não for relevante, basta responder «não» e não voltarei a escrever.",
            "Italian": "Se non è rilevante, risponda semplicemente «no» e non le scriverò più."}


def assemble(body: str, contact: Contact, spec: CampaignSpec, pack: Pack) -> str:
    """Greeting, body, sign-off, signature block and (commercial playbooks) the opt-out line — all in code."""
    sign = f"Best regards,\n{spec.sender.name}" + (f"\n{spec.sender.signature.strip()}" if spec.sender.signature.strip() else "")
    full = f"{salutation(contact, pack)}\n\n{body}\n\n{sign}".strip()
    if getattr(pack, "commercial", False):
        footer = _OPT_OUT.get(spec.language, _OPT_OUT[""])
        if spec.sender.postal_address.strip():
            footer += "\n" + spec.sender.postal_address.strip()
        full += "\n\n--\n" + footer
    return full


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
            body = body_of(d.body)  # without greeting, sign-off and footer
            blocks.append(f"### id: {c.id}\nEVIDENCE ABOUT RECIPIENT:\n{_recipient_facts(c)}\nDRAFT:\n{body}")
        user = f"SENDER FACTS:\n{sender_facts}\nGoal: {spec.objective}; timing: {spec.timeline or 'n/a'}\n\n" + "\n\n".join(blocks)
        # Careful reading matters more here than anywhere else, so the checker thinks harder.
        data = llm.json(system=guard(VERIFY_SYSTEM), user=user, purpose="verify", tier="verify", reasoning="medium",
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
    if hasattr(llm, "model_for"):  # who checked: independent of the writer, or not
        stats["verifier"] = llm.model_for("verify")
        stats["independent_of_writer"] = stats["verifier"] != llm.model_for("strong")
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
    body = body_of(d.body)
    user = (_prompt(c, spec, pack) + "\n\nA fact-checker found these claims unsupported by the facts above:\n- "
            + "\n- ".join(claims) + f"\n\nRewrite the draft without them (do not replace them with other unsupported claims).\n\nDraft:\n{body}")
    try:
        data = llm.json(system=SYSTEM, user=user, purpose="draft_fix", tier="strong", temperature=0.3, max_tokens=700)
    except LLMError as e:
        log.warning("rewrite failed for %s: %s", c.name, e)
        return None
    subject = choose_subject(_subjects(data), c, spec) or d.subject
    new_body = clean_body(str(data.get("body") or ""), c)
    issues = check(new_body, subject, c, spec)
    return Draft(contact_id=c.id, to=d.to, subject=subject, body=assemble(new_body, c, spec, pack), issues=issues)


def body_of(full: str) -> str:
    """The model-written middle of an assembled email (no greeting, sign-off, signature or footer)."""
    text = full.split("\n\n--\n", 1)[0]
    text = text.split("\n\n", 1)[1] if "\n\n" in text else text
    return text.rsplit("\n\nBest regards", 1)[0].strip()


# ── follow-ups ───────────────────────────────────────────────────────────────

FOLLOWUP_SYSTEM = """You write a short follow-up to a cold email that got no reply.
Rules: 40-80 words; do not repeat the first email; bring ONE new, true reason drawn from the evidence or the sender
facts that the first email did not use; restate the ask in one easy sentence (or offer to stop writing); no guilt
("just bumping", "did you see my email"), no greeting line, no sign-off. Use only the facts given.
Return JSON: {"body": "<text>"}"""


def followup_email(contact: Contact, spec: CampaignSpec, pack: Pack, llm: LLM, first: Draft, n: int = 1,
                   ctx=None) -> Optional[Draft]:
    """Follow-up number n (1 or 2) to a sent first email: a new angle from evidence the first email did not use."""
    spec = with_context(spec, ctx)
    first_body = body_of(first.body)
    unused = [e for e in contact.evidence if e.text[:40].lower() not in first_body.lower()][:4]
    user = (_prompt(contact, spec, pack) + f"\n\nFIRST EMAIL (sent, no reply):\n{first_body}\n\n"
            "EVIDENCE NOT USED YET:\n" + ("\n".join(f"- {e.text}" for e in unused) or "- (none: use another sender fact)")
            + f"\n\nThis is follow-up number {n}." + (" Make it the last one: offer to stop writing." if n >= 2 else ""))
    try:
        data = llm.json(system=FOLLOWUP_SYSTEM, user=user, purpose="followup", tier="strong", temperature=0.4, max_tokens=300)
    except LLMError as e:
        log.warning("follow-up failed for %s: %s", contact.name, e)
        return None
    body = clean_body(str((data or {}).get("body") or ""), contact)
    subject = first.subject if first.subject.lower().startswith("re:") else f"Re: {first.subject}"
    short = spec.model_copy(update={"word_target": 60})
    issues = [i for i in check(body, subject, contact, short) if not i.startswith(("too short", "subject"))]
    first_low = " ".join(first_body.lower().split())
    reused = [x for x in re.split(r"(?<=[.!?])\s+", body.strip()) if len(x.split()) >= 5
              and " ".join(x.lower().split()) in first_low]
    if reused:
        issues.append("repeats the first email")
    return Draft(contact_id=contact.id, to=first.to, subject=subject, body=assemble(body, contact, spec, pack), issues=issues)
