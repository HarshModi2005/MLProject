"""
Playbooks: everything that differs between kinds of outreach, kept as data.

A playbook says who the targets are (people or organisations), which discovery tools find them,
what to ask the user, which older lists would help, how to judge fit and how to write. The engine
itself is the same for every playbook.

Adapting the system to a new use case means adding a playbook — either here or as a JSON file in
the directory named by BRAIN_PLAYBOOKS_DIR (default ./playbooks), with the same field names.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List

log = logging.getLogger("brain.playbooks")

# Discovery tools a playbook may list (see brain/tools.py):
#   openalex  researchers via their publications          free
#   orgs      organisations via web search + list pages   search calls (+ LLM for list pages)
#   places    local businesses via Google Maps (Serper)   search calls, returns website + address
#   news      journalists via recent articles (Serper)    search calls + page fetches
#   github    developers via GitHub user search           free (rate-limited)
#   web       people listed on staff/team pages           search calls
#   events    sponsors / speakers of similar past events  search calls + page fetches
#   jobs      companies hiring now (public ATS boards)    search calls + free ATS APIs
#   portfolio investors whose portfolio fits the topic    search calls + page fetches
#   podcasts  podcasts via the iTunes directory + RSS     free (owner email from the feed)
#   lookalike organisations similar to the user's examples / the best finds so far   search calls
#   coauthors researchers who publish with the best finds so far (OpenAlex graph)     free
#   memory    prospects already known from earlier campaigns                          free
# The planner may add tools beyond `sources` when the campaign's fit criteria need proof that
# only those tools can produce (see brain/explore.py).


@dataclass(frozen=True)
class Playbook:
    name: str
    title: str
    description: str  # when to use it; shown to the planner when it picks a playbook
    target_kind: str = "org"  # person | org
    sources: List[str] = field(default_factory=lambda: ["orgs"])
    default_roles: List[str] = field(default_factory=list)
    reject_roles: List[str] = field(default_factory=list)
    min_seniority: float = 0.0
    salutation: str = "first_name"  # professor | first_name
    query_templates: List[str] = field(default_factory=list)  # {topic} {place} {role}
    preferred_inboxes: List[str] = field(default_factory=list)
    accept_inboxes: bool = True
    judge_rubric: str = ""
    writing_notes: str = ""
    word_target: int = 150
    upload_prompts: List[str] = field(default_factory=list)  # older lists worth asking the user for
    questions: List[str] = field(default_factory=list)  # worth asking when the prompt does not say
    # Kinds of organisation that are proper targets besides companies (see brain/discovery/precision/orgtype.py):
    # academic, media, publisher, government, nonprofit. Others are demoted before enrichment.
    allowed_org_classes: List[str] = field(default_factory=list)
    # What makes a prospect good, beyond matching the topics (a topic criterion is always added).
    # Each item: {"name", "signal", "weight", "terms", "description", "must"} — see models.FitCriterion.
    criteria: List[dict] = field(default_factory=list)
    timing: List[dict] = field(default_factory=list)  # reasons why now is a good moment, same shape
    anti_patterns: List[str] = field(default_factory=list)  # phrases that mark a poor target
    # Pre-ranking weights; missing keys fall back to the defaults for the target kind (brain/rank.py).
    rank_weights: Dict[str, float] = field(default_factory=dict)

    @property
    def sources_set(self) -> set:
        return set(self.sources)


_UPLOAD_COMMON = ["a do-not-contact list or people you have already emailed (so nobody is contacted twice)",
                  "an email you sent before that worked well (to match your style)"]

_BUILTIN: List[Playbook] = [
    Playbook(
        name="academic", allowed_org_classes=["academic"], title="Researchers and faculty",
        description="Professors, researchers or labs found through their publications: research internships, "
                    "PhD/postdoc inquiries, academic collaboration.",
        target_kind="person", sources=["openalex", "arxiv", "nsf_awards", "coauthors", "web"],
        default_roles=["professor", "assistant professor", "associate professor", "principal investigator",
                       "research scientist", "lecturer", "reader"],
        reject_roles=["phd student", "student", "research scholar", "intern"], min_seniority=0.45,
        salutation="professor", accept_inboxes=False,
        query_templates=["{topic} faculty {place}", "{topic} professor research group {place}"],
        judge_rubric="Their recent publications overlap the sender's work or stated interests, and they lead a group.",
        writing_notes="Reference one specific recent paper and connect it to something the sender actually did. "
                      "Precision over enthusiasm.", word_target=170,
        upload_prompts=["your CV", "a list of professors or labs you already like or have contacted"] + _UPLOAD_COMMON[:1],
        questions=["Which research topics exactly?", "Which countries or universities?", "When, and for how long?"],
        criteria=[{"name": "publishes on the topic recently", "signal": "publications", "weight": 1.5,
                   "terms": ["paper", "journal", "conference", "proceedings"]},
                  {"name": "leads a group", "signal": "topic", "weight": 0.8,
                   "terms": ["lab", "group", "professor", "principal investigator", "students"]}],
        timing=[{"name": "recruiting students", "signal": "hiring", "weight": 0.5,
                 "terms": ["prospective students", "openings", "positions available", "hiring", "join us"]}],
        anti_patterns=["phd student", "retired", "emeritus"]
    ),
    Playbook(
        name="sponsorship", title="Event sponsors",
        description="Companies or brands to sponsor an event, fest, hackathon, conference, club or team.",
        sources=["events", "orgs", "lookalike"], default_roles=["Brand Manager", "Marketing Manager", "Partnerships", "Head of Marketing",
                                          "Community", "Developer Relations", "Campus"],
        preferred_inboxes=["partnerships", "sponsorship", "sponsor", "marketing", "brand", "events", "community", "hello"],
        query_templates=["companies that sponsored {topic} {place}", "{topic} sponsors {place}",
                         "brands sponsoring college fests {place}"],
        judge_rubric="The company markets to the event's audience and has sponsored similar events or runs campus/"
                     "community programs. Sponsorship agencies and event-listing sites are poor targets.",
        writing_notes="Lead with the audience and reach numbers, name 2-3 concrete sponsorship options, one clear ask.",
        upload_prompts=["last year's sponsor list (to re-approach them and to find similar companies)",
                        "your sponsorship deck or brochure"] + _UPLOAD_COMMON,
        questions=["What is the audience size and profile?", "What sponsorship tiers or benefits do you offer?",
                   "Any industries or brands to avoid?"],
        criteria=[{"name": "has sponsored similar events", "signal": "past_sponsorship", "weight": 2.0,
                   "terms": ["sponsor", "sponsored", "partner", "hackathon", "fest", "summit", "title sponsor"]},
                  {"name": "markets to the event's audience", "signal": "audience", "weight": 1.0,
                   "terms": ["students", "developers", "campus", "community", "youth", "college"]}],
        timing=[{"name": "runs campus or community programs", "signal": "audience", "weight": 0.6,
                 "terms": ["campus ambassador", "student program", "developer program", "community program"]}],
        anti_patterns=["event management company", "sponsorship agency", "ticketing", "event listing"],
        rank_weights={"fit": 0.45, "sim": 0.35}
    ),
    Playbook(
        name="b2b_sales", title="B2B sales leads",
        description="Businesses that could buy a product or service: SaaS, agencies, local businesses such as "
                    "clinics, restaurants, schools or shops.",
        sources=["places", "orgs", "lookalike"], default_roles=["Owner", "Founder", "CEO", "Director", "Manager", "Head"],
        preferred_inboxes=["info", "contact", "hello", "business", "sales", "office", "appointments"],
        query_templates=["{topic} {place}", "best {topic} in {place}", "{topic} companies {place}"],
        judge_rubric="The business clearly has the problem the offer solves and is the right size to buy it.",
        writing_notes="One specific observation about their business, the offer in one sentence, a low-commitment "
                      "next step. Never promise results beyond the sender's stated facts.", word_target=120,
        upload_prompts=["your current customers (to find lookalike businesses)", "a lead list you bought or built"]
        + _UPLOAD_COMMON,
        questions=["Which kind of business, exactly?", "Which city or area?", "What is the offer and its price?"],
        criteria=[{"name": "is the kind of business that buys this", "signal": "local_business", "weight": 1.2,
                   "terms": []},
                  {"name": "right size", "signal": "topic", "weight": 0.6, "terms": ["team", "branches", "locations",
                   "customers", "clients"]}],
        timing=[{"name": "growing", "signal": "hiring", "weight": 0.6, "terms": ["hiring", "careers", "new branch",
                 "now open", "expanding"]}],
        anti_patterns=["directory", "marketplace", "franchise opportunity"]
    ),
    Playbook(
        name="job_search", title="Job and internship search",
        description="Hiring managers, founders, CTOs or recruiters at companies, to ask for a job, internship or referral.",
        sources=["orgs", "jobs", "lookalike"], default_roles=["Founder", "Co-founder", "CTO", "Engineering Manager", "Head of Engineering",
                                          "Recruiter", "Talent", "Hiring Manager"],
        preferred_inboxes=["careers", "jobs", "hiring", "talent", "founders", "hr", "hello"],
        query_templates=["{topic} startups {place} hiring", "{topic} companies {place} careers"],
        judge_rubric="The company works on what the candidate is good at and is plausibly hiring at their level.",
        writing_notes="Name the role, one or two proofs of skill relevant to their product, ask for a short call.",
        word_target=140,
        upload_prompts=["your CV", "companies you applied to already"] + _UPLOAD_COMMON[:1],
        questions=["Which role and seniority?", "Which cities or remote?", "Company stage or size preferences?"],
        criteria=[{"name": "builds what the candidate is good at", "signal": "topic", "weight": 1.2, "terms": []},
                  {"name": "hiring now", "signal": "hiring", "weight": 1.5,
                   "terms": ["hiring", "careers", "open roles", "job", "engineer", "intern", "we're hiring"]}],
        timing=[{"name": "recently raised money", "signal": "hiring", "weight": 0.6,
                 "terms": ["raised", "series a", "seed round", "funding"]}],
        anti_patterns=["staffing agency", "recruitment agency", "job board", "consultancy for hire"],
        rank_weights={"fit": 0.45, "sim": 0.3, "timing": 0.15}
    ),
    Playbook(
        name="fundraising", title="Investors",
        description="Angel investors, VC funds or accelerators to raise money for a startup.",
        sources=["portfolio", "orgs", "lookalike"], default_roles=["Partner", "Managing Partner", "Principal", "Angel", "Investor", "Founder"],
        preferred_inboxes=["pitch", "pitches", "deals", "invest", "founders", "hello", "contact"],
        query_templates=["{topic} investors {place}", "pre-seed {topic} venture capital {place}",
                         "angel investors {topic} {place}"],
        judge_rubric="The investor backs this sector and stage in this geography, ideally with relevant portfolio companies.",
        writing_notes="Traction first, the raise in one line, ask for a short call. No hype words.", word_target=130,
        upload_prompts=["investors who already passed or are in conversation", "your pitch deck"] + _UPLOAD_COMMON,
        questions=["Stage and amount being raised?", "Key traction numbers?", "Investor geography?"],
        criteria=[{"name": "invests in this sector", "signal": "portfolio_fit", "weight": 2.0,
                   "terms": ["portfolio", "invested", "backed", "investments"]},
                  {"name": "invests at this stage", "signal": "topic", "weight": 1.0,
                   "terms": ["pre-seed", "seed", "early stage", "series a", "angel"]}],
        timing=[{"name": "new fund or active this year", "signal": "recent_coverage", "weight": 0.6,
                 "terms": ["new fund", "fund ii", "fund iii", "announces", "closes"]}],
        anti_patterns=["investment banking", "mutual fund", "stock broker", "wealth management", "crypto exchange"],
        rank_weights={"fit": 0.5, "sim": 0.3}
    ),
    Playbook(
        name="recruiting", title="Recruiting candidates",
        description="Finding people to hire or invite: developers, designers, researchers, open-source contributors.",
        target_kind="person", sources=["github", "npm", "huggingface"], accept_inboxes=False,
        default_roles=["Engineer", "Developer", "Designer", "Researcher"],
        query_templates=["{topic} developer {place}", "{topic} engineer portfolio {place}"],
        judge_rubric="Their public work shows the required skills at the required level.",
        writing_notes="Mention one specific piece of their public work, the role in one sentence, an easy next step.",
        word_target=120, upload_prompts=["people already in your pipeline"] + _UPLOAD_COMMON,
        questions=["Which skills and seniority?", "Location or remote?", "What is the role?"],
        criteria=[{"name": "public work shows the skills", "signal": "open_source", "weight": 1.5,
                   "terms": ["repositories", "contributor", "maintainer", "stars"]}],
        anti_patterns=["bot", "organization account"]
    ),
    Playbook(
        name="pr_media", allowed_org_classes=["media"], title="Journalists and media",
        description="Journalists, editors, newsletter writers or podcasters, to pitch a story or get coverage.",
        target_kind="person", sources=["news", "gdelt", "podcasts"], accept_inboxes=True,
        preferred_inboxes=["news", "editor", "editorial", "tips", "press", "newsdesk", "story"],
        default_roles=["Journalist", "Reporter", "Editor", "Correspondent", "Writer", "Host"],
        query_templates=["{topic} {place}", "{topic} startup news {place}"],
        judge_rubric="They have recently written about this topic for an outlet that reaches the right readers.",
        writing_notes="Reference their recent article, the story angle in one sentence, why now, offer data or access.",
        word_target=130, upload_prompts=["your press release or announcement", "journalists who covered you before"]
        + _UPLOAD_COMMON[:1],
        questions=["What is the news?", "Which publications or beats matter?", "Embargo or timing?"],
        criteria=[{"name": "wrote about the topic recently", "signal": "recent_coverage", "weight": 2.0,
                   "terms": []}],
        timing=[{"name": "active this month", "signal": "recent_coverage", "weight": 0.6, "terms": []}],
        anti_patterns=["press release distribution", "sponsored content"]
    ),
    Playbook(
        name="partnerships", title="Partnerships and collaborations",
        description="Organisations to partner with: co-marketing, integrations, distribution, community or "
                    "NGO/CSR partnerships, university-industry collaboration.",
        sources=["orgs", "lookalike"], default_roles=["Partnerships", "Business Development", "Founder", "Head of", "Director"],
        preferred_inboxes=["partnerships", "partners", "bd", "business", "collab", "hello", "info"],
        query_templates=["{topic} organisations {place}", "{topic} companies {place}"],
        judge_rubric="Both sides gain something concrete; their audience or product complements the sender's.",
        writing_notes="State the mutual benefit concretely, propose one small first step.",
        upload_prompts=["current or past partners (to find similar ones)"] + _UPLOAD_COMMON,
        questions=["What would each side get?", "Which kinds of organisations?", "Where?"],
        criteria=[{"name": "complementary audience or product", "signal": "audience", "weight": 1.2, "terms": []},
                  {"name": "does partnerships", "signal": "past_sponsorship", "weight": 0.8,
                   "terms": ["partners", "partnership", "integrations", "collaborat", "co-marketing"]}],
        anti_patterns=["directory", "listing site"]
    ),
    Playbook(
        name="creators", allowed_org_classes=["media"], title="Creators, influencers and podcasts",
        description="YouTubers, Instagram creators, newsletter writers or podcast hosts for collaborations, "
                    "guest appearances or promotions.",
        target_kind="person", sources=["podcasts", "web", "orgs"], accept_inboxes=True,
        preferred_inboxes=["collab", "collabs", "business", "partnerships", "hello", "contact", "booking"],
        default_roles=["Creator", "Host", "Founder", "Writer"],
        query_templates=["{topic} podcast {place}", "{topic} youtuber {place}", "{topic} newsletter"],
        judge_rubric="Their audience matches and they do collaborations of this kind.",
        writing_notes="Show you know their content, the collaboration in one sentence, what they get.", word_target=120,
        upload_prompts=["creators you worked with before"] + _UPLOAD_COMMON,
        questions=["Which platform and audience size?", "What is the collaboration and budget?"],
        criteria=[{"name": "audience matches", "signal": "audience", "weight": 1.5, "terms": []},
                  {"name": "does collaborations", "signal": "topic", "weight": 0.8,
                   "terms": ["sponsor", "collab", "guest", "partnership", "brand deals", "advertise"]}],
        timing=[{"name": "published recently", "signal": "recent_coverage", "weight": 0.6, "terms": []}],
        anti_patterns=["talent agency", "mcn"]
    ),
    Playbook(
        name="speakers", allowed_org_classes=["academic", "media"], title="Speakers, judges and mentors",
        description="Experts to invite as speakers, judges, panelists, guests or mentors for an event or program.",
        target_kind="person", sources=["events", "web", "openalex", "arxiv"], accept_inboxes=False,
        default_roles=["Founder", "Director", "Head", "Professor", "Lead", "Principal"],
        query_templates=["{topic} speaker {place}", "{topic} expert {place}"],
        judge_rubric="Recognised expertise in the topic and plausible availability for this kind of invitation.",
        writing_notes="Why them specifically, the event in one line (date, audience), what is asked of them.",
        word_target=140, upload_prompts=["past speakers or judges"] + _UPLOAD_COMMON[:1],
        questions=["Event date, format and audience?", "Honorarium or travel covered?"],
        criteria=[{"name": "recognised expertise", "signal": "topic", "weight": 1.2, "terms": []},
                  {"name": "has spoken at similar events", "signal": "spoke_before", "weight": 1.2,
                   "terms": ["speaker", "keynote", "panel", "talk", "judge", "mentor"]}],
        anti_patterns=["speakers bureau", "booking agency"]
    ),
    Playbook(
        name="customer_research", title="Customer interviews and beta users",
        description="People or businesses to interview, survey or recruit as beta testers or early customers.",
        sources=["places", "orgs"], default_roles=["Owner", "Founder", "Manager", "Head"],
        preferred_inboxes=["hello", "info", "contact"],
        query_templates=["{topic} {place}", "{topic} companies {place}"],
        judge_rubric="They plausibly have the problem being researched.",
        writing_notes="Make clear nothing is being sold, the time asked (e.g. 20 minutes), what they get in return.",
        word_target=110, upload_prompts=["people who signed up or showed interest"] + _UPLOAD_COMMON[:1],
        questions=["Who exactly has the problem?", "What do participants get?"],
        criteria=[{"name": "has the problem", "signal": "topic", "weight": 1.5, "terms": []}],
        anti_patterns=["directory", "marketplace"]
    ),
    Playbook(
        name="suppliers", title="Suppliers and vendors",
        description="Suppliers, manufacturers, vendors or service providers to buy from or get quotes from.",
        sources=["places", "orgs"], default_roles=["Sales", "Owner", "Manager"],
        preferred_inboxes=["sales", "info", "enquiry", "enquiries", "contact", "orders"],
        query_templates=["{topic} supplier {place}", "{topic} manufacturer {place}"],
        judge_rubric="They supply exactly what is needed at the needed scale in the needed region.",
        writing_notes="Specify quantity, timeline and requirements; ask for a quote.", word_target=110,
        upload_prompts=["current suppliers (to avoid or compare)"],
        questions=["What exactly, how much, by when?", "Where should they be based?"],
        criteria=[{"name": "supplies exactly this", "signal": "topic", "weight": 1.5,
                   "terms": ["manufacturer", "supplier", "wholesale", "exporter", "bulk", "moq"]}],
        anti_patterns=["marketplace", "directory", "trading platform"]
    ),
    Playbook(
        name="grants", allowed_org_classes=["nonprofit", "government"], title="Grants and CSR funding",
        description="Foundations, CSR arms of companies or grant programmes to fund an NGO, research or social project.",
        sources=["orgs", "lookalike"], default_roles=["Program Officer", "CSR Head", "Program Director", "Trustee",
                                                       "Head of Sustainability", "Grants Manager"],
        preferred_inboxes=["grants", "csr", "foundation", "programs", "partnerships", "info"],
        query_templates=["{topic} grants {place}", "CSR funding {topic} {place}", "foundations funding {topic} {place}"],
        judge_rubric="The funder supports this cause in this geography, at this size, and has an open or recurring cycle.",
        writing_notes="The problem and evidence of impact first, the amount and use of funds in one line, one clear ask.",
        word_target=150, upload_prompts=["funders who supported you or similar organisations"] + _UPLOAD_COMMON[:1],
        questions=["What cause and geography?", "How much funding, for what?", "Registered entity type (e.g. 12A/80G, 501(c)(3))?"],
        criteria=[{"name": "funds this cause", "signal": "funding_focus", "weight": 2.0,
                   "terms": ["grant", "grantees", "funded", "csr", "foundation", "programme", "focus areas"]}],
        timing=[{"name": "open call", "signal": "funding_focus", "weight": 0.8,
                 "terms": ["call for proposals", "applications open", "deadline", "rfp"]}],
        anti_patterns=["grant writing services", "consultancy", "loan"],
    ),
    Playbook(
        name="general", title="General outreach",
        description="Anything else: organisations or people to contact for a purpose not covered above.",
        sources=["orgs"], default_roles=["Founder", "Director", "Head", "Manager"],
        preferred_inboxes=["hello", "contact", "info", "partnerships"],
        query_templates=["{topic} {place}", "{topic} organisations {place}"],
        judge_rubric="The target is plausibly able and willing to grant the ask.",
        writing_notes="Why this recipient, the ask in one sentence, easy to say yes. No flattery, no buzzwords.",
        upload_prompts=["any list of people or organisations related to this"] + _UPLOAD_COMMON,
        questions=["Who exactly should receive this?", "What is the ask?"],
    ),
]

_FIELD_NAMES = {f.name for f in fields(Playbook)}


def _load_user_playbooks() -> Dict[str, Playbook]:
    folder = Path(os.getenv("BRAIN_PLAYBOOKS_DIR", "playbooks"))
    out: Dict[str, Playbook] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else [data]
            for item in items:
                pb = Playbook(**{k: v for k, v in item.items() if k in _FIELD_NAMES})
                out[pb.name] = pb
        except (ValueError, TypeError) as e:
            log.warning("ignoring playbook file %s: %s", path, e)
    return out


PLAYBOOKS: Dict[str, Playbook] = {pb.name: pb for pb in _BUILTIN}
PLAYBOOKS.update(_load_user_playbooks())


def get_playbook(name: str) -> Playbook:
    return PLAYBOOKS.get(name) or PLAYBOOKS["general"]


def catalog_text() -> str:
    """The playbook menu shown to the planner LLM."""
    return "\n".join(f'- "{pb.name}" ({pb.target_kind} targets): {pb.description}' for pb in PLAYBOOKS.values())


def to_dict(pb: Playbook) -> dict:
    return asdict(pb)
