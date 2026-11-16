"""
Turning what the user says into a CampaignSpec — in ONE LLM call.

The legacy flow spent a call per slot plus confirmation, planning and explanation calls. Here the
model fills the whole spec at once and lists what is genuinely missing; anything the user already
passed explicitly (flags) is applied in code afterwards and never left to the model.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import List, Optional, Tuple

from brain.llm import LLM
from brain.models import CampaignSpec, SenderProfile, TargetPersona
from brain.packs import PACKS, get_pack
from brain.textutil import split_locations, squash

SPEC_SYSTEM = """You configure a cold-outreach campaign from a user's description.
Return JSON with exactly these keys:
{
 "pack": "academic" ONLY if the targets are researchers/faculty to be found through their publications; for companies,
   clinics, startups, investors, sponsors, recruiters or any other organisation use "general",
 "objective": "<what the sender wants, one sentence>",
 "ask": "<the concrete request to make in the email, one sentence>",
 "persona": {"description": "<who to reach, one sentence>", "roles": ["<job titles that qualify>"], "org_types": ["<kinds of organisation>"]},
 "topics": ["<3-6 specific subject areas the targets should work on; noun phrases a search engine would match>"],
 "expanded_terms": ["<4-8 synonyms, sub-areas or methods closely related to the topics>"],
 "locations": ["<countries or 'Region, Country' exactly as the user stated; empty if worldwide>"],
 "organizations": ["<specific organisations the user named, if any>"],
 "exclude_organizations": [],
 "timeline": "<when, or null>",
 "tone": "professional" | "friendly" | "concise",
 "search_queries": ["<6-10 web search queries. For 'general': queries that surface the TARGET ORGANISATIONS' own
   websites or curated lists of them (e.g. 'dental clinic Indiranagar Bengaluru', 'top edtech startups India 2026',
   'companies that sponsored college tech fests India'). For 'academic': queries that surface faculty/team pages.
   Never use site:linkedin.com or other sites that require login.>"],
 "preferred_inboxes": ["<for 'general': shared-inbox names that suit this ask, best first, e.g. partnerships, sponsorship,
   marketing for sponsorship; info, appointments for a clinic; careers, founders for jobs; pitch, deals for investors>"],
 "missing": ["<plain-language questions for information that is essential and absent; empty if none>"]
}
Only derive topics from what the user wrote or what the sender profile shows. Do not broaden the geography."""

SENDER_SYSTEM = """Extract a sender profile for cold outreach from the text (a CV or self-description).
Curate; do not dump. Return JSON:
{"name": "", "headline": "<role/degree and institution or company, under 15 words>",
 "summary": "<3-4 factual sentences>",
 "highlights": ["<6-10 items, each formatted '<where>: <what>' and under 40 words. <where> is the employer or
   organisation name for work done in a job (one item per distinct piece of work), or exactly 'Project', 'Research',
   'Award' or 'Education'. For technical work ALWAYS keep the specific methods, models and tools named in the text
   (e.g. 'LSTM, Transformer, attention'), since they are what makes the sender relevant to a particular recipient>"],
 "skills": ["<up to 12>"], "links": ["<urls present in the text>"]}
Use only facts present in the text."""


def sender_from_text(text: str, llm: LLM, attachment_path: Optional[str] = None) -> SenderProfile:
    data = llm.json(system=SENDER_SYSTEM, user=text[:9000], purpose="sender_profile", tier="strong", max_tokens=900)
    profile = SenderProfile(
        name=str(data.get("name") or ""), headline=str(data.get("headline") or ""),
        summary=str(data.get("summary") or ""),
        highlights=[str(h) for h in (data.get("highlights") or [])][:7],
        skills=[str(s) for s in (data.get("skills") or [])][:12],
        links=[str(l) for l in (data.get("links") or [])][:5],
        attachment_path=attachment_path,
    )
    return profile


def sender_from_pdf(path: str, llm: LLM) -> SenderProfile:
    import fitz  # PyMuPDF

    with fitz.open(path) as doc:
        text = "\n".join(page.get_text() for page in doc)
    if len(text.strip()) < 100:
        raise ValueError(f"Could not read text from {path} (is it a scanned PDF?)")
    # The LLM result is cached by content, so re-running with the same CV costs nothing.
    return sender_from_text(text, llm, attachment_path=path)


def new_campaign_id(seed: str = "") -> str:
    return hashlib.sha1((seed or uuid.uuid4().hex).encode()).hexdigest()[:12]


def build_spec(description: str, sender: SenderProfile, llm: LLM, *, target_count: int = 25,
               locations: Optional[List[str]] = None, pack: Optional[str] = None,
               answers: str = "") -> Tuple[CampaignSpec, List[str]]:
    """One LLM call. Returns (spec, questions the user still needs to answer)."""
    user = f"USER REQUEST:\n{description}\n"
    if answers:
        user += f"\nADDITIONAL ANSWERS FROM USER:\n{answers}\n"
    user += f"\nSENDER PROFILE:\n{sender.facts_text()[:2500]}"
    data = llm.json(system=SPEC_SYSTEM, user=user, purpose="campaign_spec", tier="strong", max_tokens=1100)

    countries, regions = split_locations(locations if locations else [str(x) for x in data.get("locations") or []])
    persona = data.get("persona") or {}
    chosen_pack = pack or (data.get("pack") if data.get("pack") in PACKS else "general")
    spec = CampaignSpec(
        id=new_campaign_id(),
        pack=chosen_pack,
        objective=squash(str(data.get("objective") or description), 300),
        ask=squash(str(data.get("ask") or ""), 300),
        persona=TargetPersona(
            description=squash(str(persona.get("description") or ""), 300),
            roles=[str(r) for r in persona.get("roles") or []][:8],
            org_types=[str(r) for r in persona.get("org_types") or []][:5],
        ),
        topics=[str(t) for t in data.get("topics") or []][:6],
        expanded_terms=[str(t) for t in data.get("expanded_terms") or []][:8],
        countries=countries, regions=regions,
        organizations=[str(o) for o in data.get("organizations") or []][:15],
        exclude_organizations=[str(o) for o in data.get("exclude_organizations") or []][:15],
        search_queries=[str(q) for q in data.get("search_queries") or []][:10],
        preferred_inboxes=[str(q).strip().lower() for q in data.get("preferred_inboxes") or []][:8],
        timeline=(str(data["timeline"]) if data.get("timeline") else None),
        tone=str(data.get("tone") or "professional"),
        target_count=target_count,
        sender=sender,
    )
    questions = [str(q) for q in data.get("missing") or []][:4]
    if not spec.topics:
        questions.append("Which topics or fields should the people you contact be working on?")
    return spec, questions


def spec_from_fields(*, topics: List[str], objective: str, sender: SenderProfile, locations: Optional[List[str]] = None,
                     pack: str = "academic", roles: Optional[List[str]] = None, ask: str = "",
                     target_count: int = 25, timeline: Optional[str] = None) -> CampaignSpec:
    """Build a spec with no LLM call at all (flags / API use)."""
    countries, regions = split_locations(locations or [])
    spec = CampaignSpec(
        id=new_campaign_id(), pack=pack, objective=objective, ask=ask or objective,
        persona=TargetPersona(description=objective, roles=roles or []),
        topics=topics, countries=countries, regions=regions, target_count=target_count,
        timeline=timeline, sender=sender,
    )
    spec.search_queries = fallback_queries(spec)
    return spec


def fallback_queries(spec: CampaignSpec, limit: int = 8) -> List[str]:
    """Template-built web queries, used when the spec was built without the LLM."""
    pack = get_pack(spec.pack)
    places = spec.regions or spec.organizations[:3] or [""]
    roles = spec.persona.roles[:1] or pack.default_roles[:1] or [""]
    out: List[str] = []
    for topic in spec.topics[:4]:
        for place in places[:2]:
            for tpl in pack.query_templates:
                q = squash(tpl.format(topic=topic, place=place, role=roles[0]))
                if q not in out:
                    out.append(q)
    return out[:limit]
