"""Organisation-first campaigns (sponsors, B2B leads, jobs, investors) against a fake mini-internet.
No network, no API keys: search results, web pages and LLM replies are all scripted."""

import json

import httpx

from brain.cache import Cache
from brain.fetch import Fetcher
from brain.llm import LLM
from brain.models import CampaignSpec, SenderProfile, TargetPersona
from brain.orgs import _inbox_rank, _org_name_from_title, is_aggregator
from brain.pipeline import Deps, run_campaign
from brain.settings import Settings
from brain.sources.orcid import Orcid
from brain.sources.websearch import Hit, WebSearch
from brain.store import Store

SITES = {
    # A clinic in Bengaluru with only a shared inbox and an address on its contact page.
    "https://smiledental.in/": """<html><head><title>Smile Dental Clinic | Best dentist in Indiranagar</title>
        <meta name="description" content="Family dental clinic offering implants, braces and root canal treatment.">
        </head><body><nav><a href="/contact">Contact</a><a href="/blog/x">Blog</a></nav><h1>Smile Dental Clinic</h1>
        <p>Gentle, modern dentistry for the whole family since 2009.</p></body></html>""",
    "https://smiledental.in/contact": """<html><title>Contact | Smile Dental</title><body><h1>Contact us</h1>
        <p>12 CMH Road, Indiranagar, Bengaluru 560038</p><p>Write to info@smiledental.in or call us.</p>
        <p>Website issues: webmaster@smiledental.in</p></body></html>""",
    # A clinic in Mumbai: must be dropped for a Bangalore campaign.
    "https://bandraphysio.com/": """<html><title>Bandra Physio</title><body><h1>Bandra Physio</h1>
        <p>Sports physiotherapy in Bandra West, Mumbai. Email hello@bandraphysio.com</p></body></html>""",
    # A company with a team page naming the right person.
    "https://devtoolco.io/": """<html><title>DevToolCo - APIs for developers</title><head>
        <meta property="og:site_name" content="DevToolCo"></head><body>
        <nav><a href="/team">Team</a><a href="/privacy">Privacy</a></nav>
        <p>DevToolCo builds developer APIs used by 40,000 student developers across India, Bengaluru HQ.</p>
        <p>General: hello@devtoolco.io</p></body></html>""",
    "https://devtoolco.io/team": """<html><title>Team | DevToolCo</title><body><h1>Our team</h1>
        <div><h3><a href="/team/priya">Priya Nair</a></h3><p>Head of Partnerships</p>
        <a href="mailto:priya@devtoolco.io">email</a></div>
        <div><h3><a href="/team/arjun">Arjun Rao</a></h3><p>CTO</p><a href="mailto:arjun@devtoolco.io">email</a></div>
        <div><h3><a href="/team/kiran">Kiran Das</a></h3><p>Designer</p></div>
        <div><h3><a href="/team/meera">Meera Shah</a></h3><p>Engineer</p></div></body></html>""",
    # A list page naming an organisation that it does not link to.
    "https://inc42.com/top-edtech": """<html><title>Top 10 edtech startups in India</title><body><h1>Top edtech startups</h1>
        <p>1. LearnLoop — AI tutoring for school students, Bengaluru.</p>
        <p>2. DevToolCo — developer APIs. <a href="https://devtoolco.io/">site</a></p></body></html>""",
    "https://learnloop.in/": """<html><title>LearnLoop</title><body><h1>LearnLoop</h1>
        <p>AI tutoring for Class 9-12 students, built in Bengaluru. Partnerships: partnerships@learnloop.in,
        support@learnloop.in</p></body></html>""",
    # Only a no-reply address: nobody to write to.
    "https://ghostco.in/": """<html><title>GhostCo</title><body><h1>GhostCo</h1>
        <p>Bengaluru software studio. noreply@ghostco.in</p></body></html>""",
}

SEARCH = {
    "q": [Hit("https://smiledental.in/", "Smile Dental Clinic | Best dentist in Indiranagar", "dental clinic Bengaluru"),
          Hit("https://bandraphysio.com/", "Bandra Physio", "physiotherapy clinic"),
          Hit("https://devtoolco.io/", "DevToolCo - APIs for developers", "developer tools Bengaluru"),
          Hit("https://inc42.com/top-edtech", "Top 10 edtech startups in India", "list"),
          Hit("https://ghostco.in/", "GhostCo", "software studio Bengaluru")],
    '"LearnLoop" Bangalore official website': [Hit("https://learnloop.in/", "LearnLoop", "AI tutoring")],
}


def router(req: httpx.Request) -> httpx.Response:
    url = str(req.url)
    if url.endswith("/robots.txt"):
        return httpx.Response(404)
    html = SITES.get(url)
    return httpx.Response(200, text=html, headers={"content-type": "text/html"}) if html else httpx.Response(404)


class FakeSearch(WebSearch):
    def search(self, query, n=8):
        self.calls += 1
        return SEARCH.get(query, [])


class ScriptedLLM:
    def __init__(self):
        self.seen = []

    def __call__(self, model, system, user, temperature, max_tokens):
        self.seen.append(system[:30])
        if system.startswith("You read a web page and list"):
            return json.dumps({"organisations": [
                {"name": "LearnLoop", "website": None, "about": "AI tutoring for school students", "location": "Bengaluru"},
                {"name": "DevToolCo", "website": "https://devtoolco.io/", "about": "developer APIs"},
                {"name": "Made Up Corp", "website": "https://madeup.ai/"},  # not on the page: must be ignored
            ]})
        if system.startswith("You plan web searches"):
            return json.dumps({"queries": []})
        if system.startswith("You assess"):
            cards = json.loads(user.split("Candidates:\n", 1)[1])
            return json.dumps({"results": [{"id": c["id"], "score": 80, "reasons": ["fits"], "hook_ref": 0} for c in cards]})
        if system.startswith("You write"):
            body = ("Your clinic and team caught my attention because of the work described on your website. "
                    "We run a small service and would like to offer you a short free trial so you can judge it "
                    "yourselves, with no obligation and no setup work on your side. It takes a few minutes to start, "
                    "and we handle everything else for you, including onboarding your staff and answering questions. "
                    "If it does not help, you simply stop using it. Would you be open to a short call next week "
                    "to see whether it is a fit for you?")
            return json.dumps({"subject": "A short free trial for your team", "body": body})
        if system.startswith("You fact-check"):
            ids = [l.split("id: ", 1)[1].strip() for l in user.splitlines() if l.startswith("### id: ")]
            return json.dumps({"results": [{"id": i, "unsupported": []} for i in ids]})
        raise AssertionError(system[:60])


def make(tmp_path):
    st = Settings()
    st.serper_api_key, st.database_url, st.per_host_delay = "x", f"sqlite:///{tmp_path}/g.db", 0
    cache = Cache()
    client = httpx.Client(transport=httpx.MockTransport(router))
    script = ScriptedLLM()
    deps = Deps(settings=st, cache=cache, llm=LLM(st, cache, transport=script), fetcher=Fetcher(st, cache, client=client),
                search=FakeSearch(st, cache, client=client), orcid=Orcid(st, cache, client=client), openalex=None,
                store=Store(st.database_url))
    return deps, script


def spec(**kw):
    base = dict(id="g1", pack="general", objective="Offer a free trial", ask="A 15-minute call",
                persona=TargetPersona(description="decision makers", roles=["Head of Partnerships", "Founder", "Owner"]),
                topics=["dental clinic", "developer tools", "edtech"], countries=["IN"], regions=["Bangalore"],
                search_queries=["q"], preferred_inboxes=["partnerships", "info"], target_count=10,
                sender=SenderProfile(name="Harsh Modi", summary="Runs a two-person agency."))
    base.update(kw)
    return CampaignSpec(**base)


def test_org_first_campaign_end_to_end(tmp_path):
    deps, script = make(tmp_path)
    result = run_campaign(spec(), deps)
    got = {c.name: (c.kind, c.best_email.address if c.best_email else None) for c in result.shortlist}

    # Named person with a fitting role beats the company's hello@ inbox.
    assert got["Priya Nair"] == ("person", "priya@devtoolco.io")
    # Clinic with no named owner: its shared inbox, never webmaster@.
    assert got["Smile Dental Clinic"] == ("org", "info@smiledental.in")
    # Found only by name on a list page; website resolved by one search; preferred inbox chosen over support@.
    assert got["LearnLoop"] == ("org", "partnerships@learnloop.in")
    # Mumbai clinic dropped on location (Bengaluru alias of Bangalore kept the others), GhostCo has nobody to write to,
    # the list publisher is never a target, and the LLM's invented organisation was ignored.
    assert set(got) == {"Priya Nair", "Smile Dental Clinic", "LearnLoop"}
    assert result.funnel["dropped_after_reading_sites"] == {"region": 1}

    drafts = {c.name: result.drafts[c.id] for c in result.shortlist}
    assert drafts["Smile Dental Clinic"].body.startswith("Hello Smile Dental Clinic team,")
    assert drafts["Priya Nair"].body.startswith("Hi Priya,")
    assert result.funnel["cost"]["llm"]["by_purpose"]["extract_orgs"] == 1


def test_inbox_choice_and_helpers():
    pref = ["partnerships", "marketing"]
    assert _inbox_rank("partnerships@x.com", pref) < _inbox_rank("info@x.com", pref)
    assert _inbox_rank("noreply@x.com", pref) is None and _inbox_rank("privacy@x.com", pref) is None
    assert _inbox_rank("press@x.com", ["press"]) == 0  # allowed when the campaign asks for it
    assert is_aggregator("wellfound.com") and is_aggregator("www.practo.com") and not is_aggregator("devtoolco.io")
    assert _org_name_from_title("Smile Dental Clinic | Best dentist in Indiranagar", "smiledental.in") == "Smile Dental Clinic"
    assert _org_name_from_title("Home - Acme Robotics", "acmerobotics.com") == "Acme Robotics"
    # Real failures: a question and a section heading were used as company names.
    assert _org_name_from_title("What's in it for colleges?", "collegebrandconnect.com") == "Collegebrandconnect"
    assert _org_name_from_title("Sponsorships | Tata group", "tata.com") == "Tata group"
    assert _org_name_from_title("Sponsorships", "tata.com") == "Tata"
