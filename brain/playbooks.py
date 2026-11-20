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
    # Accept clearly-flagged guessed addresses (an organisation's usual convention) as reachable
    # (brain/discovery/reach/patterns.py). Sensible where people expect cold mail at work addresses.
    accept_guessed: bool = False
    # Catalog grouping, so near-neighbours are compared side by side when a playbook is chosen.
    family: str = "other"
    # Commercial outreach (selling, partnerships for revenue): emails get an opt-out line (draft.py).
    commercial: bool = False
    # Words that describe the SENDER's side (their event, the role they seek), not what targets are: topics made of
    # them are moved out of the search topics (brain/intake.py), since discovery searches for what targets ARE.
    topic_stoplist: List[str] = field(default_factory=list)
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
        name="academic", family="research", allowed_org_classes=["academic"], title="Researchers and faculty",
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
        name="sponsorship", topic_stoplist=["fest", "festival", "hackathon", "event", "sponsor", "conference", "summit", "college", "student", "corporate"], family="funding", title="Event sponsors",
        description="Companies or brands to sponsor an event, fest, hackathon, conference, club or team.",
        sources=["events", "orgs", "lookalike", "sponsor_walk", "graph_walk"], default_roles=["Brand Manager", "Marketing Manager", "Partnerships", "Head of Marketing",
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
        name="b2b_sales", family="selling", commercial=True, title="B2B sales leads",
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
        name="job_search", topic_stoplist=["internship", "intern", "job", "role", "leadership", "career", "hiring", "position"], family="jobs", accept_guessed=True, title="Job and internship search",
        description="Hiring managers, founders, CTOs or recruiters at companies, to ask for a job, internship or referral.",
        sources=["orgs", "jobs", "hn_hiring", "lookalike"], default_roles=["Founder", "Co-founder", "CTO", "Engineering Manager", "Head of Engineering",
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
        name="fundraising", family="funding", title="Investors",
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
        name="recruiting", family="people", title="Recruiting candidates",
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
        name="pr_media", family="media", accept_guessed=True, allowed_org_classes=["media"], title="Journalists and media",
        description="Journalists, editors, newsletter writers or podcasters, to pitch a story or get coverage.",
        target_kind="person", sources=["news", "gdelt", "podcasts", "beat_walk", "graph_walk"], accept_inboxes=True,
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
        name="partnerships", family="selling", commercial=True, title="Partnerships and collaborations",
        description="Organisations to partner with: co-marketing, integrations, distribution, community or "
                    "NGO/CSR partnerships, university-industry collaboration.",
        sources=["orgs", "lookalike", "graph_walk"], default_roles=["Partnerships", "Business Development", "Founder", "Head of", "Director"],
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
        name="creators", family="media", allowed_org_classes=["media"], title="Creators, influencers and podcasts",
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
        name="speakers", topic_stoplist=["summit", "conference", "event", "speaker", "talk"], family="people", allowed_org_classes=["academic", "media"], title="Speakers, judges and mentors",
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
        name="customer_research", family="selling", commercial=True, title="Customer interviews and beta users",
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
        name="suppliers", family="buying", title="Suppliers and vendors",
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
        name="grants", family="funding", allowed_org_classes=["nonprofit", "government"], title="Grants and CSR funding",
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
    # ── discovery v3: more use cases (docs/discovery-v3-plan.md §6) ──────────────────────────────────────────
    Playbook(
        name="cofounder", family="people", title="Co-founder or technical partner",
        description="A co-founder or early technical partner who has built things like what the sender is starting.",
        target_kind="person", sources=["github", "npm", "huggingface", "events", "web"], accept_inboxes=False,
        default_roles=["Engineer", "Founder", "Developer", "Researcher"],
        query_templates=["{topic} hackathon winners {place}", "{topic} side project builder {place}"],
        judge_rubric="They have shipped work close to the idea and show signs of being able to commit (indie, between roles, building).",
        writing_notes="What you are building and why now, what you bring, why them specifically. Suggest a 20-minute call.",
        questions=["What are you building?", "What skills are you missing?", "Where should they be based?"],
        criteria=[{"name": "has built something close to the idea", "signal": "open_source", "weight": 1.5, "terms": []},
                  {"name": "won or built at hackathons", "signal": "spoke_before", "weight": 0.6,
                   "terms": ["hackathon", "winner", "built"]}],
        anti_patterns=["recruiter", "agency", "consultancy", "hiring manager"],
    ),
    Playbook(
        name="advisors", family="people", title="Advisors and board members",
        description="Experienced operators or investors to advise a company or join its board.",
        target_kind="person", sources=["orgs", "news", "events", "web"], accept_inboxes=False,
        default_roles=["Founder", "CEO", "CTO", "VP", "Partner", "Director"], min_seniority=0.6,
        query_templates=["former {topic} founder {place}", "{topic} industry veteran speaker {place}"],
        judge_rubric="They operated (founded, led, scaled) in this space and their experience covers the sender's next hard problem.",
        writing_notes="The specific problem you want advice on, why their experience fits, a small first ask (one call).",
        questions=["What problem do you want advice on?", "Paid, equity or informal?"],
        criteria=[{"name": "operated in this space", "signal": "topic", "weight": 1.5, "terms": []},
                  {"name": "senior and recognised", "signal": "spoke_before", "weight": 0.8, "terms": ["keynote", "founder", "board"]}],
        anti_patterns=["coach for hire", "consultancy", "agency"],
    ),
    Playbook(
        name="accelerators", family="funding", title="Accelerators, incubators and programmes",
        description="Accelerators, incubators, fellowships or startup programmes to apply to or partner with.",
        sources=["orgs", "events", "lookalike"], allowed_org_classes=["academic", "nonprofit", "government"],
        default_roles=["Program Manager", "Partner", "Director", "Head of Programs"],
        preferred_inboxes=["apply", "programs", "hello", "info", "partnerships"],
        query_templates=["{topic} accelerator {place}", "{topic} startup incubator {place} applications", "{topic} fellowship founders {place}"],
        judge_rubric="The programme backs this sector and stage, in this geography, and has an open or upcoming cohort.",
        writing_notes="Traction in one line, why this programme, a precise question (fit, timeline) or the application.",
        questions=["Sector and stage?", "Equity-free only?"],
        criteria=[{"name": "backs this sector and stage", "signal": "portfolio_fit", "weight": 1.5,
                   "terms": ["cohort", "portfolio", "alumni", "startups"]}],
        timing=[{"name": "applications open", "signal": "funding_focus", "weight": 0.8, "terms": ["apply", "applications open", "deadline"]}],
        anti_patterns=["coworking space only", "consultancy", "paid course"],
    ),
    Playbook(
        name="distributors", family="selling", commercial=True, title="Distributors, dealers and resellers",
        description="Distributors, dealers, resellers or channel partners to sell a product in a region.",
        sources=["places", "orgs", "events", "lookalike"],
        default_roles=["Owner", "Director", "Business Development", "Purchase Head", "Channel Manager"],
        preferred_inboxes=["sales", "business", "purchase", "info", "contact"],
        query_templates=["{topic} distributor {place}", "{topic} dealers {place}", "{topic} wholesale {place}"],
        judge_rubric="They already sell adjacent products to the same buyers in this region and are large enough to stock it.",
        writing_notes="Margin and support in one line, why their catalogue fits, a call or a sample as the ask.",
        questions=["Which product and region?", "Margins and minimum order?"],
        criteria=[{"name": "sells adjacent products in the region", "signal": "local_business", "weight": 1.5, "terms": []},
                  {"name": "exhibits at trade shows", "signal": "exhibits_at", "weight": 0.6, "terms": ["expo", "exhibitor"]}],
        anti_patterns=["manufacturer only", "retail shop", "directory"],
    ),
    Playbook(
        name="government_tenders", family="selling", commercial=True, title="Government buyers and tenders",
        description="Government departments, agencies or public bodies that buy what the sender offers, or have open tenders.",
        sources=["orgs", "news"], allowed_org_classes=["government"],
        default_roles=["Director", "Joint Secretary", "Procurement Officer", "Commissioner", "Program Officer"],
        preferred_inboxes=["procurement", "tenders", "info", "contact"],
        query_templates=["{topic} tender {place}", "{topic} government procurement {place}", "{topic} request for proposal {place}"],
        judge_rubric="The body buys this kind of product or service, in this jurisdiction, and has a current or recurring need.",
        writing_notes="Formal and brief: the capability, relevant past public work, compliance facts, the ask (meeting or tender clarification).",
        questions=["What do you offer?", "Which jurisdiction?", "Registrations (GeM, SAM, MSME)?"],
        criteria=[{"name": "buys this", "signal": "open_tender", "weight": 1.5, "terms": ["tender", "procurement", "rfp", "bid"]}],
        anti_patterns=["tender aggregator", "consultancy", "news portal"],
    ),
    Playbook(
        name="integrations", family="selling", commercial=True, title="Integration and technology partners",
        description="Products or platforms to integrate with, list on, or co-market with (APIs, marketplaces, app stores).",
        sources=["orgs", "lookalike", "google_play"], default_roles=["Partnerships", "Head of Partnerships", "Developer Relations", "Product Manager"],
        preferred_inboxes=["partners", "partnerships", "integrations", "bd", "hello"],
        query_templates=["{topic} integrations marketplace", "{topic} technology partner program", "{topic} API partners"],
        judge_rubric="Their users would benefit from the integration and they run a partner or marketplace programme.",
        writing_notes="The shared user and the concrete integration in one line, what you have already built, a small next step.",
        questions=["What does your product do?", "Which platforms do your users already use?"],
        criteria=[{"name": "runs a partner programme", "signal": "topic", "weight": 1.2, "terms": ["partner program", "marketplace", "integrations", "api"]}],
        anti_patterns=["agency", "reseller only", "directory"],
    ),
    Playbook(
        name="seo_guest_posts", family="media", title="Guest posts and backlinks",
        description="Blogs, newsletters and publications in a niche that accept guest posts or link to useful resources.",
        target_kind="org", sources=["orgs", "news"], allowed_org_classes=["media"],
        default_roles=["Editor", "Content Manager", "Founder"], preferred_inboxes=["editor", "contribute", "hello", "contact"],
        query_templates=["{topic} \"write for us\"", "{topic} \"guest post\" guidelines", "{topic} blog contributors"],
        judge_rubric="The site is real and active in the niche, publishes contributions or resource links, and is not a link farm.",
        writing_notes="Two specific post ideas that fit their audience, one line on your expertise, no link-exchange offers.",
        questions=["What topics can you write about?", "Do you have published samples?"],
        criteria=[{"name": "accepts contributions", "signal": "topic", "weight": 1.2, "terms": ["write for us", "guest post", "contribute"]}],
        anti_patterns=["paid links", "link farm", "pbn", "casino"],
    ),
    Playbook(
        name="brand_deals", family="media", commercial=True, title="Brand deals for creators",
        description="Brands that sponsor creators, podcasts or newsletters like the sender's.",
        sources=["orgs", "podcasts", "lookalike"], default_roles=["Brand Manager", "Influencer Marketing", "Partnerships", "Marketing Manager"],
        preferred_inboxes=["partnerships", "brand", "marketing", "collab", "hello"],
        query_templates=["{topic} brand sponsorships creators", "{topic} influencer marketing campaign {place}", "{topic} podcast sponsors"],
        judge_rubric="The brand markets to the creator's audience and already sponsors creators or shows of a similar size.",
        writing_notes="Audience size and fit in numbers, one past collaboration, a concrete package and rate range.",
        questions=["Audience size and demographics?", "Channels (YouTube, podcast, newsletter)?"],
        criteria=[{"name": "sponsors similar creators", "signal": "past_sponsorship", "weight": 1.5,
                   "terms": ["sponsored by", "creator", "influencer", "ambassador"]},
                  {"name": "markets to this audience", "signal": "audience", "weight": 1.0, "terms": []}],
        anti_patterns=["influencer agency", "affiliate network", "marketplace"],
    ),
    Playbook(
        name="campus_recruiting", topic_stoplist=["hiring", "fresher", "recruitment drive"], family="jobs", title="Campus recruiting and student communities",
        description="Placement cells, career offices and student clubs at colleges, to recruit or run programmes with students.",
        sources=["orgs", "places", "events"], allowed_org_classes=["academic"],
        default_roles=["Training and Placement Officer", "Placement Officer", "Career Services", "Club Lead", "Dean"],
        preferred_inboxes=["placement", "placements", "tpo", "careers", "careerservices", "info"],
        query_templates=["{topic} college placement cell {place}", "engineering colleges {place} training and placement", "{topic} student club {place}"],
        judge_rubric="The institution trains students in the relevant field, in this region, and its placement office or club is reachable.",
        writing_notes="Roles and timeline, eligibility, what students get. Offer a pre-placement talk.",
        questions=["Which roles and how many?", "Which regions or colleges?"],
        criteria=[{"name": "trains students in this field", "signal": "topic", "weight": 1.5, "terms": ["engineering", "department", "students"]}],
        anti_patterns=["coaching institute", "online course seller", "admission consultancy"],
    ),
    Playbook(
        name="school_outreach", family="local", title="Schools and educators",
        description="Schools, principals and teachers, to pilot an education product or programme.",
        sources=["places", "orgs"], allowed_org_classes=["academic", "nonprofit", "government"],
        default_roles=["Principal", "Head of School", "Academic Coordinator", "Teacher"],
        preferred_inboxes=["principal", "office", "admin", "info", "contact"],
        query_templates=["schools in {place}", "{topic} school {place}", "CBSE schools {place}"],
        judge_rubric="A real school in the area teaching the right grades, plausibly open to a pilot.",
        writing_notes="What students gain, how little time it asks of teachers, a free pilot as the ask. Respectful and brief.",
        questions=["Which grades and subjects?", "Free pilot or paid?"],
        criteria=[{"name": "a school teaching these grades", "signal": "local_business", "weight": 1.5, "terms": ["school", "grade", "class"]}],
        anti_patterns=["tuition centre", "coaching", "edtech company", "directory"],
    ),
    Playbook(
        name="phd_positions", family="research", title="PhD and research positions",
        description="Professors with funded projects who may take PhD students, postdocs or research interns.",
        target_kind="person", salutation="professor", sources=["openalex", "arxiv", "nsf_awards", "coauthors", "web"],
        allowed_org_classes=["academic"], accept_inboxes=False,
        default_roles=["Professor", "Associate Professor", "Assistant Professor", "Principal Investigator"],
        query_templates=["{topic} PhD positions {place}", "{topic} lab openings {place}", "{topic} professor {place} research group"],
        judge_rubric="Recent work matches the sender's interests, they lead a group, and there are signs of funding or openings.",
        writing_notes="One paper of theirs, read properly; one result of yours; the precise position and start date.",
        questions=["Research interests?", "Start date and funding needs?"],
        criteria=[{"name": "publishes on the topic recently", "signal": "publications", "weight": 1.5, "terms": []},
                  {"name": "has funding", "signal": "grant_awarded", "weight": 1.0, "terms": ["grant", "funded", "award"]}],
        timing=[{"name": "recruiting students", "signal": "hiring", "weight": 0.8, "terms": ["phd position", "openings", "join my group"]}],
        anti_patterns=["retired", "emeritus", "adjunct"],
    ),
    Playbook(
        name="awards_cfps", family="funding", title="Awards, competitions and calls for proposals",
        description="Organisers of awards, competitions, grants-for-startups or conference calls the sender could enter.",
        sources=["orgs", "events", "news"], allowed_org_classes=["academic", "nonprofit", "government", "media"],
        default_roles=["Program Manager", "Organiser", "Editor", "Director"],
        preferred_inboxes=["awards", "submissions", "programs", "hello", "info"],
        query_templates=["{topic} awards {place} nominations", "{topic} competition {place} apply", "{topic} call for proposals"],
        judge_rubric="The call is open or recurring, fits the sender's work, and the organiser is reachable for a question.",
        writing_notes="A precise eligibility question or the submission itself. Never ask to be shortlisted.",
        questions=["What would you enter?", "Deadlines that matter?"],
        criteria=[{"name": "an open or recurring call", "signal": "funding_focus", "weight": 1.5,
                   "terms": ["nominations", "apply", "deadline", "call for"]}],
        anti_patterns=["paid award", "pay to enter", "directory"],
    ),
    Playbook(
        name="volunteering", family="local", title="NGOs and volunteering",
        description="NGOs and community organisations to volunteer with, partner with or offer pro-bono help to.",
        sources=["orgs", "places"], allowed_org_classes=["nonprofit"],
        default_roles=["Founder", "Program Manager", "Volunteer Coordinator", "Director"],
        preferred_inboxes=["volunteer", "volunteers", "contact", "info", "hello"],
        query_templates=["{topic} NGO {place}", "{topic} volunteer {place}", "{topic} non-profit {place}"],
        judge_rubric="An active organisation working on this cause in this area that could use what the sender offers.",
        writing_notes="What you can give and for how long, why this cause. Make it easy to say yes.",
        questions=["Which cause and area?", "Time or skills you offer?"],
        criteria=[{"name": "works on this cause here", "signal": "funding_focus", "weight": 1.5, "terms": []}],
        anti_patterns=["fundraising platform", "directory", "csr consultancy"],
    ),
    Playbook(
        name="alumni", family="people", title="Alumni networking",
        description="Alumni of the sender's school or past employer working at target organisations or in target roles.",
        target_kind="person", sources=["web", "orgs", "news"], accept_inboxes=False,
        default_roles=["Engineer", "Manager", "Director", "Founder"],
        query_templates=["{topic} alumni {place}", "\"{topic}\" alumnus engineer"],
        judge_rubric="A real alum (school or employer stated in the evidence) in a role relevant to the ask.",
        writing_notes="The shared school or employer first, a specific question about their path, a 15-minute call.",
        questions=["Which school or past employer?", "Which companies or roles?"],
        criteria=[{"name": "shares the sender's school or employer", "signal": "warm_path", "weight": 1.5, "terms": ["alumni", "alumnus"]}],
        anti_patterns=["recruiter", "staffing agency"],
    ),
    Playbook(
        name="oss_maintainers", family="people", title="Open-source maintainers and developers",
        description="Maintainers or contributors of open-source libraries, tools or packages (Rust crates, npm, Python, Go...) "
                    "to ask for API feedback, a review, adoption, a contribution or a collaboration.",
        target_kind="person", sources=["github", "npm", "crates", "web"], accept_inboxes=False,
        default_roles=["Maintainer", "Author", "Core contributor", "Engineer"],
        query_templates=["{topic} crate maintainer", "{topic} library maintainer github"],
        judge_rubric="A named person who maintains or leads a real, actively used open-source project on the topic "
                     "(the evidence shows the project and their role in it).",
        writing_notes="Name their project and one specific thing about it, say what you built in one sentence, "
                      "ask for a small review or a look at a specific API, and make it cheap to say no.",
        upload_prompts=["a list of the libraries or repositories you depend on or compete with"],
        questions=["Which library or project is yours (link)?", "What exactly would you like from the maintainer?"],
        anti_patterns=["recruiter", "staffing agency", "course", "bootcamp"],
    ),
    Playbook(
        name="general", family="other", title="General outreach",
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


FAMILY_TITLES = {"research": "Research and academia", "people": "Individual people", "media": "Media and creators",
                 "funding": "Money and programmes", "jobs": "Jobs and hiring", "selling": "Selling and partnerships",
                 "buying": "Buying", "local": "Local and community", "other": "Anything else"}


def catalog_text() -> str:
    """The playbook menu shown to the planner LLM, grouped by family so near-neighbours are compared side by side."""
    out = []
    for fam, title in FAMILY_TITLES.items():
        pbs = [pb for pb in PLAYBOOKS.values() if pb.family == fam]
        if not pbs:
            continue
        out.append(f"{title}:")
        out += [f'  - "{pb.name}" ({pb.target_kind} targets): {pb.description}' for pb in pbs]
    return "\n".join(out)


def to_dict(pb: Playbook) -> dict:
    return asdict(pb)
