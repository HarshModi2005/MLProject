"""Discovery v3 phase 3, source wave 1: arXiv (+ PDF emails), NSF awards, npm, Hugging Face, GDELT, Google Play,
more job boards, and the fetcher / template features they rely on. Synthetic fixtures (made-up people), no network."""

import json

import httpx

from brain.cache import Cache
from brain.discovery.normalize import add_published_email, render
from brain.discovery.reach.pdfemails import emails_in, expand_grouped, match_authors
from brain.fetch import Fetcher, Page
from brain.models import Contact
from brain.settings import Settings
from brain.sources.websearch import Hit
from brain.tools import TOOLS
from tests.brain.test_discovery_v3 import Deps, FakeFetcher, FakeSearch, _run, _spec

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2611.00001v1</id><published>2026-11-02T10:00:00Z</published>
<title>Message Passing for Molecular Graphs</title><summary>We study message passing networks.</summary>
<author><name>Alice Rao</name><arxiv:affiliation>Indian Institute of Science</arxiv:affiliation></author>
<author><name>Bharat Mehta</name></author><author><name>Chen Wei</name></author>
<link title="pdf" href="https://arxiv.org/pdf/2611.00001v1" rel="related" type="application/pdf"/></entry>
</feed>"""


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_rate_limit_sleeps(monkeypatch):
    monkeypatch.setattr("brain.discovery.policy.time.sleep", lambda s: None)


class SourceFetcher(FakeFetcher):
    def __init__(self, pages=None, api=None, texts=None, pdfs=None):
        super().__init__(pages, api)
        self.texts, self.pdfs, self.pdf_calls = texts or {}, pdfs or {}, []

    def get_json(self, url, params=None, ttl=0, headers=None):
        self.json_calls.append((url, dict(params or {}), dict(headers or {})))
        return self.api.get(url)

    def get_api_text(self, url, params=None, ttl=0):
        self.json_calls.append((url, dict(params or {}), {}))
        return self.texts.get(url)

    def get_pdf_text(self, url, max_pages=2):
        self.pdf_calls.append(url)
        return self.pdfs.get(url)


# ── shared pieces ────────────────────────────────────────────────────────────
def test_template_prefix_and_default_filters():
    assert render('"{t}"{c|prefix= sourcecountry:}', {"t": "ml", "c": "india"}) == '"ml" sourcecountry:india'
    assert render('"{t}"{c|prefix= sourcecountry:}', {"t": "ml", "c": ""}) == '"ml"'
    assert render("{x|default=US}", {}) == "US"


def test_published_emails_skip_unread_inboxes_and_keep_org_inboxes():
    org, person = Contact(kind="org", name="Studio"), Contact(kind="person", name="Asha Iyer")
    assert not add_published_email(org, "mailto:support@studio.in", 0.8, None)
    assert not add_published_email(org, "npm-oidc-no-reply@github.com", 0.8, None)
    assert not add_published_email(org, "support+android@studio.in", 0.8, None)
    assert add_published_email(org, "mailto:hello@studio.in?subject=hi", 0.8, None)
    assert org.best_email.address == "hello@studio.in" and org.best_email.method == "inbox"
    assert not add_published_email(person, "hello@studio.in", 0.8, None)  # a person is not a shared inbox
    assert add_published_email(person, "asha@studio.in", 0.8, None) and person.best_email.method == "published"


def test_pdf_emails_expand_groups_and_match_authors_by_name():
    text = "Alice Rao, Bharat Mehta, Chen Wei\n{alice.rao, bmehta}@iisc.ac.in  chen@xyz.edu  info@iisc.ac.in"
    assert expand_grouped(text) == ["alice.rao@iisc.ac.in", "bmehta@iisc.ac.in"]
    assert "info@iisc.ac.in" not in emails_in(text)
    assert match_authors(text, ["Alice Rao", "Bharat Mehta", "Chen Wei"]) == {
        "Alice Rao": "alice.rao@iisc.ac.in", "Bharat Mehta": "bmehta@iisc.ac.in", "Chen Wei": "chen@xyz.edu"}
    assert match_authors("a.kumar@x.in", ["Anil Kumar", "Arun Kumar"]) == {}  # ambiguous: nobody


# ── sources ──────────────────────────────────────────────────────────────────
def test_arxiv_keeps_only_authors_whose_paper_prints_their_address():
    f = SourceFetcher(texts={"https://export.arxiv.org/api/query": ATOM},
                      pdfs={"https://arxiv.org/pdf/2611.00001v1": "Alice Rao, Bharat Mehta\n{alice.rao, bmehta}@iisc.ac.in"})
    found, stats = TOOLS["arxiv"].fn(_run(_spec(), Deps(f), pb="academic"))
    assert sorted(c.name for c in found) == ["Alice Rao", "Bharat Mehta"] and stats["pdfs_read"] == 1
    alice = next(c for c in found if c.name == "Alice Rao")
    assert alice.best_email.address == "alice.rao@iisc.ac.in" and alice.best_email.method == "pdf"
    assert alice.country == "IN" and alice.org == "Indian Institute of Science" and alice.signals["latest_year"] == 2026
    assert alice.evidence[0].text == "Message Passing for Molecular Graphs" and not any(fl.startswith("pdf:") for fl in alice.flags)
    url, params, _ = f.json_calls[0]
    assert params["search_query"] == 'all:"speech recognition"' and params["max_results"] == 10


def test_nsf_awards_give_investigators_with_their_listed_address():
    award = {"id": "2600001", "title": "Graph learning for materials", "piFirstName": "DANA", "piLastName": "SMITH",
             "piEmail": "dsmith@state.edu", "awardeeName": "STATE UNIVERSITY", "awardeeCity": "AUSTIN",
             "awardeeStateCode": "TX", "startDate": "09/01/2026", "estimatedTotalAmt": "450000", "abstractText": "We study graphs."}
    f = SourceFetcher(api={"https://api.nsf.gov/services/v1/awards.json": {"response": {"award": [award, dict(award)]}}})
    found, _ = TOOLS["nsf_awards"].fn(_run(_spec(countries=["US"]), Deps(f), pb="academic"))
    assert len(found) == 1
    c = found[0]
    assert c.name == "Dana Smith" and c.org == "State University" and c.country == "US" and c.location_text == "Austin, TX"
    assert c.best_email.address == "dsmith@state.edu" and c.signals["grant_awarded"] == 1.0
    assert c.evidence[0].detail.startswith("$450,000")
    assert f.json_calls[0][1]["dateStart"].count("/") == 2  # three years back, US date format


def test_npm_maintainers_with_registry_addresses():
    pkg = {"package": {"name": "indic-asr", "description": "Speech recognition for Indian languages", "version": "1.2.0",
                       "date": "2026-10-01T00:00:00Z", "keywords": ["asr", "speech"],
                       "publisher": {"email": "npm-oidc-no-reply@github.com", "actor": {"name": "asha", "email": "asha@dev.in"}},
                       "maintainers": [{"username": "asha", "email": "asha@dev.in"}],
                       "links": {"npm": "https://www.npmjs.com/package/indic-asr", "repository": "git+https://github.com/ashai/indic-asr.git"}}}
    f = SourceFetcher(api={"https://registry.npmjs.org/-/v1/search": {"objects": [pkg]}})
    found, _ = TOOLS["npm"].fn(_run(_spec(), Deps(f), pb="recruiting"))
    c = found[0]
    assert c.name == "asha" and c.best_email.address == "asha@dev.in" and "https://github.com/ashai" in c.urls
    assert c.signals["maintains_package"] == 1.0 and c.topics == ["asr", "speech"]
    bot = {"package": {"name": "@types/dom-speech", "maintainers": [{"username": "types", "email": "ts-npm-types@microsoft.com"}]}}
    f = SourceFetcher(api={"https://registry.npmjs.org/-/v1/search": {"objects": [bot]}})
    assert TOOLS["npm"].fn(_run(_spec(), Deps(f), pb="recruiting"))[0] == []


def test_huggingface_authors_merge_across_models():
    models = [{"modelId": "voxlab/asr-hi", "downloads": 120000, "likes": 30, "pipeline_tag": "automatic-speech-recognition",
               "tags": ["transformers", "hi", "license:mit"]}, {"modelId": "voxlab/asr-ta", "downloads": 5000, "likes": 2}]
    f = SourceFetcher(api={"https://huggingface.co/api/models": models})
    found, _ = TOOLS["huggingface"].fn(_run(_spec(), Deps(f), pb="recruiting"))
    assert [c.name for c in found] == ["voxlab"] and len(found[0].evidence) == 2
    assert found[0].urls == ["https://huggingface.co/voxlab"] and found[0].topics == ["transformers", "hi"]


def test_gdelt_finds_journalists_without_spending_searches():
    art = """<html><head><meta name="author" content="Meera Pillai"></head><body>Story</body></html>"""
    api = {"https://api.gdeltproject.org/api/v2/doc/doc": {"articles": [
        {"url": "https://paper.in/a1", "title": "Indic speech AI", "domain": "paper.in", "seendate": "20261101T000000Z"},
        {"url": "https://paper.in/a1", "title": "dupe", "domain": "paper.in"}]}}
    f = SourceFetcher(pages={"https://paper.in/a1": art}, api=api)
    deps = Deps(f)
    found, _ = TOOLS["gdelt"].fn(_run(_spec(), deps))
    assert [c.name for c in found] == ["Meera Pillai"] and found[0].signals["recent_coverage"] == 1.0
    assert deps.search.calls == 0
    q = f.json_calls[0][1]["query"]
    assert q == '"speech recognition" sourcelang:english sourcecountry:india'


def test_google_play_developer_from_app_page():
    page = """<html><script type="application/ld+json">{"@type": "SoftwareApplication", "name": "QuickClinic",
    "applicationCategory": "MEDICAL", "description": "Book clinic visits",
    "author": {"@type": "Person", "name": "Book clinic visits fast", "url": "https://bytestudio.in"},
    "aggregateRating": {"ratingValue": "4.4", "ratingCount": "1200"}}</script>
    <a href="/store/apps/developer?id=Byte+Studio+Pvt"><span>Byte Studio</span></a>
    <a href="mailto:support@bytestudio.in">x</a><a href="mailto:ravi@bytestudio.in">y</a></html>"""
    url = "https://play.google.com/store/apps/details?id=in.byte.quick"
    web = {'site:play.google.com/store/apps/details "speech recognition"': [Hit(url, "QuickClinic - Apps on Google Play")]}
    f = SourceFetcher(pages={url: page})
    found, _ = TOOLS["google_play"].fn(_run(_spec(target_kind="org"), Deps(f, FakeSearch(web=web)), pb="b2b_sales"))
    c = found[0]
    assert c.name == "Byte Studio" and c.kind == "org" and c.org_domain == "bytestudio.in"
    assert c.best_email.address == "ravi@bytestudio.in"  # support@ is skipped, the next published address is used
    assert c.signals["app_publisher"] == 1.0 and c.signals["reviews"] == 1200.0 and "rated 4.4" in c.evidence[0].text
    from brain.discovery.contracts import Record
    from brain.discovery.hooks.google_play import to_contact
    from brain.discovery.registry import SPECS

    stuffed = to_contact(SPECS["google_play"], Record(source="google_play", data={
        "developer": "Doctor Appointment, Consultation, Meds, Tests&amp;more", "website": "https://www.practo.com"}), None)
    assert stuffed.name == "Practo" and stuffed.org_domain == "practo.com"


def test_router_enables_google_play_only_when_a_criterion_needs_app_makers():
    from brain.discovery.router import choose
    from brain.fit import profile_for
    from brain.models import FitCriterion, ProspectProfile
    from brain.playbooks import get_playbook

    pb = get_playbook("b2b_sales")
    plain = _spec(pack="b2b_sales", target_kind="org")
    assert "google_play" not in choose(plain, pb, profile_for(plain, pb), has_seeds=False)[0]
    apps = _spec(pack="b2b_sales", target_kind="org", profile=ProspectProfile(
        criteria=[FitCriterion(name="publishes apps", signal="app_publisher")]))
    assert "google_play" in choose(apps, pb, profile_for(apps, pb), has_seeds=False)[0]
    us_only = _spec(pack="academic")
    assert "nsf_awards" not in choose(us_only, get_playbook("academic"), profile_for(us_only, get_playbook("academic")),
                                      has_seeds=False)[0]


def test_more_job_boards_are_read():
    from brain.channels import ats_token, fetch_board

    api = {"https://apply.workable.com/api/v1/widget/accounts/voxa": {"name": "Voxa AI", "jobs": [
        {"title": "ML Engineer", "city": "Bengaluru", "country": "India", "url": "https://apply.workable.com/j/1",
         "published_on": "2026-11-01", "telecommuting": False}]},
           "https://voxa.recruitee.com/api/offers/": {"offers": [{"title": "Data Scientist", "location": "Bengaluru, India",
                                                                   "careers_url": "https://voxa.recruitee.com/o/ds",
                                                                   "company_name": "Voxa AI", "published_at": "2026-11-02 10:00"}]},
           "https://api.smartrecruiters.com/v1/companies/Voxa/postings": {"content": [
               {"id": "77", "name": "CV Engineer", "location": {"fullLocation": "Bengaluru, India"}, "company": {"name": "Voxa AI"},
                "releasedDate": "2026-11-03T00:00:00Z"}]}}
    f = SourceFetcher(api=api)
    assert fetch_board(f, "workable", "voxa") == ("Voxa AI", [{"title": "ML Engineer", "location": "Bengaluru, India",
                                                               "url": "https://apply.workable.com/j/1", "date": "2026-11-01"}])
    assert fetch_board(f, "recruitee", "voxa")[1][0]["location"] == "Bengaluru, India"
    assert fetch_board(f, "smartrecruiters", "Voxa")[1][0]["url"] == "https://jobs.smartrecruiters.com/Voxa/77"
    assert ats_token("https://apply.workable.com/j/ABC") is None


# ── fetcher ──────────────────────────────────────────────────────────────────
def _pdf_bytes(text: str) -> bytes:
    import fitz

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_fetcher_reads_pdf_text_honours_crawl_delay_and_replays(monkeypatch):
    pdf = _pdf_bytes("Alice Rao alice.rao@iisc.ac.in")
    seen = []

    def handler(req):
        seen.append(str(req.url))
        if req.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nCrawl-delay: 4\nAllow: /\n", headers={"content-type": "text/plain"})
        return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})

    sleeps = []
    monkeypatch.setattr("brain.fetch.time.sleep", lambda s: sleeps.append(s))
    st, cache = Settings(), Cache()
    f = Fetcher(st, cache, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert "alice.rao@iisc.ac.in" in f.get_pdf_text("https://papers.x/p/1")
    f.get_pdf_text("https://papers.x/p/2")
    assert sleeps and max(sleeps) > 3.0  # the second request to the host waited for the Crawl-delay
    st2 = Settings()
    st2.replay = "replay"
    replay = Fetcher(st2, cache, client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    assert "alice.rao" in replay.get_pdf_text("https://papers.x/p/1") and replay.get_pdf_text("https://papers.x/p/9") is None
