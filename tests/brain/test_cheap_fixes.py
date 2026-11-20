"""Cheap deterministic fixes from docs/ISSUES.md: headings as people, address and inbox talk in the body, junk greetings."""

from __future__ import annotations

import pytest

from brain.draft import check, leaks, salutation
from brain.models import CampaignSpec, Contact, EmailCandidate, is_service_address
from brain.packs import get_pack
from brain.textutil import looks_like_person_name


@pytest.mark.parametrize("name", ["Corporate Governance", "Partner Spotlight", "Frontline Workplace", "At SIM",
                                  "Investor Relations", "Sustainability Report"])
def test_headings_are_not_people(name):
    assert not looks_like_person_name(name)


@pytest.mark.parametrize("name", ["Priya Nair", "RAJ KUMAR", "K Sandeep", "Anna van der Berg", "Subhankar Mishra",
                                  "An Nguyen", "Hadas Or", "Sandeep Kumar"])
def test_real_names_still_pass(name):
    assert looks_like_person_name(name)


def test_service_desks_are_not_kept_as_addresses():
    c = Contact(kind="person", name="X Y")
    c.add_email(EmailCandidate(address="frontline-service@teamviewer.com", confidence=0.85, method="text"))
    c.add_email(EmailCandidate(address="support@x.com", confidence=0.85, method="text"))
    assert c.emails == []
    c.add_email(EmailCandidate(address="partners@x.com", confidence=0.7, method="inbox"))
    assert [e.address for e in c.emails] == ["partners@x.com"]
    assert not is_service_address("dev.sharma@x.com") and not is_service_address("kumar@service-now.com")


@pytest.mark.parametrize("text,count", [
    ("Writing to drbracesandfaces@gmail.com hoping this reaches your team.", 2),
    ("I am writing to the team inbox at Voltah, hoping this message reaches you.", 1),
    ("Please forward this to the right person on your shared mailbox.", 1),
    ("I would love your view on this. Could you point me to whoever handles it?", 0),
    ("Our SDK adds 40 KB. Would your team try it for two months?", 0),
])
def test_leaks(text, count):
    assert len(leaks(text)) == count


def test_check_flags_a_leak():
    spec = CampaignSpec(id="c1", objective="o", ask="a")
    c = Contact(kind="org", name="Voltah", org="Voltah")
    body = ("Voltah makes inverters. We build things too, and we would like to talk about how we could work together "
            "this year in Gujarat. Writing to info@thevoltah.com hoping this reaches you. Would you take a short call?")
    assert any("email address" in i for i in check(body, "Hello", c, spec))


def test_junk_decision_maker_is_not_greeted():
    pack = get_pack("grants")
    org = Contact(kind="org", name="E-Class", decision_maker={"name": "At SIM", "title": "Program Officer"})
    assert salutation(org, pack) == "Hello E-Class team,"
    org.decision_maker = {"name": "Priya Nair", "title": "Program Officer", "confidence": 0.9}
    assert salutation(org, pack) == "Hi Priya,"


# ── issues I-01, I-04, I-05, I-07, I-09 ──────────────────────────────────────

def test_low_confidence_decision_maker_gets_the_team_greeting():
    pack = get_pack("grants")
    org = Contact(kind="org", name="Voltah", decision_maker={"name": "Helen Pappas", "title": "Owner", "confidence": 0.6})
    assert salutation(org, pack) == "Hello Voltah team,"
    org.decision_maker = {"name": "Helen Pappas", "title": "Owner"}  # old record: no confidence, not a paid provider
    assert salutation(org, pack) == "Hello Voltah team,"
    org.decision_maker = {"name": "Helen Pappas", "title": "Owner", "source": "hunter"}
    assert salutation(org, pack) == "Hi Helen,"


def test_decision_maker_search_rules():
    from brain.orgs import people_from_serp_scored
    from brain.sources.websearch import Hit

    roles = ["Owner", "Director"]
    insta = [Hit("https://www.instagram.com/p/C89h/", "Afsaneh Rangiani - Owner - Dr Braces", "Dr Braces clinic owner")]
    assert people_from_serp_scored(insta, "Dr Braces", roles) is None  # a post is not a record of who works where
    dup = [Hit("https://in.linkedin.com/in/d", "Delgado Javier Javier - Director - Photovoltaic Solar | LinkedIn")]
    assert people_from_serp_scored(dup, "Photovoltaic Solar", roles) is None
    generic = [Hit("https://in.linkedin.com/in/e", "Maria Lopez - Director, Photovoltaic Engineer | LinkedIn")]
    found = people_from_serp_scored(generic, "Photovoltaic Solar", roles)
    assert found is None or found[3] < 0.7  # shares only generic words with the organisation
    good = [Hit("https://in.linkedin.com/in/y", "Priya Nair - Director - DevToolCo | LinkedIn")]
    assert people_from_serp_scored(good, "DevToolCo", roles)[3] >= 0.7


def test_directories_are_aggregators():
    from brain.textutil import is_aggregator

    assert is_aggregator("Topstartups", "topstartups.io") and is_aggregator("Top 50 AI startups in India")
    assert is_aggregator("Clutch", "https://clutch.co/in") and not is_aggregator("Valiance Solutions", "valiancesolutions.com")


def test_judge_caps_wrong_region_and_competitors_and_skips_directories():
    import json

    from brain.cache import Cache
    from brain.judge import judge
    from brain.llm import LLM
    from brain.settings import Settings

    spec = CampaignSpec(id="c", objective="Find investors", ask="a call", countries=["IN"])
    a = Contact(id="a", kind="person", name="Mark Grovic", org="US Fund", org_domain="usfund.com")
    b = Contact(id="b", kind="org", name="Voltah", org="Voltah", org_domain="thevoltah.com")
    d = Contact(id="d", kind="org", name="Topstartups", org="Topstartups", org_domain="topstartups.io")
    ok = Contact(id="e", kind="person", name="Priya Nair", org="Indian VC", org_domain="ivc.in")
    answer = json.dumps({"results": [
        {"id": "a", "score": 80, "region_ok": False, "competitor_or_directory": False, "reasons": []},
        {"id": "b", "score": 85, "region_ok": True, "competitor_or_directory": True, "reasons": []},
        {"id": "e", "score": 80, "region_ok": True, "competitor_or_directory": False, "reasons": []}]})
    llm = LLM(Settings(), Cache(), transport=lambda *a: answer)
    judge([a, b, d, ok], spec, llm)
    assert a.score <= 0.4 and "wrong_region" in a.flags
    assert b.score <= 0.25 and "competitor_or_directory" in b.flags
    assert d.score == 0 and "aggregator" in d.flags
    assert ok.score >= 0.7


def test_free_mail_is_not_an_attributed_person_address():
    from brain.discovery.reach.tiers import contact_tier

    c = Contact(kind="person", name="Bhaskarjyoti Das")
    c.add_email(EmailCandidate(address="bhaskarjyoti01@gmail.com", confidence=0.8, method="published"))
    assert contact_tier(c) == "guess"
    c2 = Contact(kind="person", name="Bhaskarjyoti Das")
    c2.add_email(EmailCandidate(address="bhaskarjyoti@pes.edu", confidence=0.8, method="published"))
    assert contact_tier(c2) == "attributed"


def test_prefer_person_over_inbox_of_same_org():
    from brain.rank import prefer_person

    inbox = Contact(id="1", kind="org", name="Valiance", org="Valiance", org_domain="valiance.com", score=0.64)
    ceo = Contact(id="2", kind="person", name="Vikas Kamra", org="Valiance", org_domain="valiance.com", score=0.60)
    assert [c.id for c in prefer_person([inbox, ceo])] == ["1"]  # the better-scoring one stays
    ceo.score = 0.70
    assert [c.id for c in prefer_person([inbox, ceo])] == ["2"]


def test_professor_title_only_with_evidence():
    pack = get_pack("academic")
    assert salutation(Contact(kind="person", name="Swagatam Das", title="Professor"), pack) == "Dear Professor Das,"
    assert salutation(Contact(kind="person", name="Subhankar Mishra"), pack) == "Dear Subhankar Mishra,"
    assert salutation(Contact(kind="person", name="A B", title="Researcher"), pack) == "Dear Dr. B,"


def test_org_named_like_a_person_is_not_greeted_through_a_shared_inbox():
    pack = get_pack("b2b_sales")
    org = Contact(kind="org", name="Om Happy Teeth")
    org.add_email(EmailCandidate(address="info@omhappyteeth.com", confidence=0.7, method="inbox"))
    assert salutation(org, pack) == "Hello Om Happy Teeth team,"
    assert salutation(Contact(kind="org", name="Gaurav Singh"), pack) == "Hi Gaurav,"  # no shared inbox: the owner's own


def test_site_builders_are_not_organisations():
    from brain.textutil import is_aggregator

    assert is_aggregator("Weebly", "weebly.com") and is_aggregator("Khoja Dental", "khojadental.com") is False
    assert is_aggregator("x", "weebly.com")


def test_inbox_rule_keeps_the_main_ask():
    import brain.draft as dr

    rules = [v for v in vars(dr).values() if isinstance(v, dict) and "org_inbox" in v][0]
    assert "never replace the main ask" in rules["org_inbox"]


@pytest.mark.parametrize("address", ["notificacionesjudiciales@unipiloto.edu.co", "merchant@atome.sg", "legal@x.com"])
def test_legal_and_merchant_desks_are_service_addresses(address):
    assert is_service_address(address)


# ── stage-judge findings: valid targets lost because no website was resolved ─────────────────────────────────

class _FakeSearch:
    def __init__(self, hits):
        self.hits, self.queries = hits, []

    def search(self, q, n=5, raw=False):
        self.queries.append(q)
        return self.hits


def _enricher(hits):
    from brain.orgs import OrgEnricher
    from brain.settings import Settings

    return OrgEnricher(fetcher=None, search=_FakeSearch(hits), settings=Settings())


def test_acronym_domain_is_resolved_from_the_result_title():
    from brain.sources.websearch import Hit

    e = _enricher([Hit("https://unal.edu.co/", "Universidad Nacional de Colombia - Sede Bogotá", "")])
    org = Contact(kind="org", name="Universidad Nacional de Colombia")
    e._resolve_site(org, CampaignSpec(id="c"))
    assert org.org_domain == "unal.edu.co"


def test_directory_and_social_results_are_not_the_website():
    from brain.sources.websearch import Hit

    hits = [Hit("https://www.justdial.com/Jaipur/Sony-Fashion", "Sony Fashion - Justdial", ""),
            Hit("https://www.facebook.com/sonyfashion", "Sony Fashion", "")]
    e = _enricher(hits)
    org = Contact(kind="org", name="Sony Fashion")
    e._resolve_site(org, CampaignSpec(id="c"))
    assert org.org_domain is None


def test_resolution_cap_follows_the_campaign_size():
    from brain.packs import get_pack

    e = _enricher([])
    e.enrich_all([], CampaignSpec(id="c", target_count=30), get_pack("general"))
    assert e.max_resolves == 80 or e.max_resolves >= 30 * 3


def test_size_limits_are_read_from_the_request():
    from brain.intake import size_from_text

    assert size_from_text("fintechs with 50 to 500 employees who might hire") == (50, 500)
    assert size_from_text("companies with over 1,000 employees") == (1000, None)
    assert size_from_text("startups under 20 people") == (None, 20)
    assert size_from_text("find clinics in Pune") == (None, None)


def test_search_provider_failures_are_reported_not_swallowed():
    import httpx

    from brain.cache import Cache
    from brain.settings import Settings
    from brain.sources.websearch import WebSearch

    def handler(request):
        return httpx.Response(400, json={"message": "Not enough credits", "statusCode": 400})

    st = Settings()
    st.serper_api_key = "k"
    ws = WebSearch(st, Cache(), httpx.Client(transport=httpx.MockTransport(handler)))
    for i in range(4):
        assert ws.search(f"query {i}") == []
    msg = ws.provider_problem()
    assert "4 of 4" in msg and "Not enough credits" in msg and "not because nobody matches" in msg
    ok = WebSearch(st, Cache(), httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"organic": []}))))
    ok.search("fine query")
    assert ok.provider_problem() == ""


def test_public_bodies_as_targets_use_the_government_playbook():
    from brain.intake import public_buyer_playbook

    water = {"persona": {"description": "Decision makers responsible for water management in municipal corporations or water boards"}}
    assert public_buyer_playbook("b2b_sales", water) == "government_tenders"
    clinics = {"persona": {"description": "Owners of independent dental clinics"}}
    assert public_buyer_playbook("b2b_sales", clinics) == "b2b_sales"
    assert public_buyer_playbook("investors", water) == "investors"  # only generic sales is redirected


def test_each_requested_place_gets_a_share_and_empty_places_are_reported():
    from brain.rank import in_location, location_gaps, location_targets, reserve_locations

    spec = CampaignSpec(id="c", regions=["Kerala", "Tamil Nadu"], countries=["IN"])
    assert location_targets(spec) == ["Kerala", "Tamil Nadu"]
    assert location_targets(CampaignSpec(id="c", regions=["Pune"], countries=["IN"])) == []
    ker = [Contact(id=f"k{i}", kind="org", name=f"K{i}", location_text="Kochi, Kerala", score=0.9 - i / 100) for i in range(8)]
    tn = [Contact(id="t1", kind="org", name="T1", location_text="Chennai, Tamil Nadu", score=0.7)]
    ordered = reserve_locations(ker + tn, ["Kerala", "Tamil Nadu"], 8, 0.6)
    assert "t1" in [c.id for c in ordered[:8]]
    assert location_gaps(ker, ["Kerala", "Tamil Nadu"]) == ["Tamil Nadu"]
    assert in_location(Contact(name="x", country="NZ"), "NZ") and not in_location(Contact(name="x", country="AU"), "NZ")


def test_a_size_limit_in_an_also_clause_is_not_applied_to_everyone():
    from brain.intake import size_from_text

    text = "small installers in Kerala, and also large EPC firms with over 500 employees"
    assert size_from_text(text) == (None, None)


# ── reach: a person's own site, its contact page, and search snippets ───────────────────────────────────────────

class _Page:
    def __init__(self, url, html="", ok=True):
        self.url, self.html, self.ok, self.status, self.error = url, html, ok, 200 if ok else 0, ""


class _FakeFetcher:
    def __init__(self, pages):
        self.pages = pages

    def get(self, url):
        return self.pages.get(url) or _Page(url, ok=False)


def _person_enricher(pages, hits=()):
    from brain.enrich import Enricher
    from brain.settings import Settings

    class S(_FakeSearch):
        available = True

    class O:
        def person(self, _):
            return [], []

    return Enricher(_FakeFetcher(pages), S(list(hits)), O(), Settings())


def test_address_on_a_personal_site_is_taken_even_if_it_is_not_a_profile_page():
    home = "https://priyanair.dev/"
    html = ('<html><head><title>Home</title></head><body><p>Notes on Rust and async runtimes, written slowly.</p>'
            '<a href="mailto:priya@priyanair.dev">email me</a></body></html>')
    e = _person_enricher({home: _Page(home, html)})
    c = Contact(kind="person", name="Priya Nair", urls=[home])
    e._enrich_one(c)
    assert c.best_email and c.best_email.address == "priya@priyanair.dev"


def test_contact_page_one_link_from_the_home_page_is_followed():
    home, contact = "https://dev.example.org/", "https://dev.example.org/contact"
    h1 = '<html><title>Home</title><body><p>Hi.</p><a href="/contact">Contact</a></body></html>'
    h2 = '<html><title>Contact</title><body><p>Write to <a href="mailto:sam.ortiz@example.org">Sam</a></p></body></html>'
    e = _person_enricher({home: _Page(home, h1), contact: _Page(contact, h2)})
    c = Contact(kind="person", name="Sam Ortiz", urls=[home])
    e._enrich_one(c)
    assert c.best_email and c.best_email.address == "sam.ortiz@example.org"


def test_a_platform_page_is_not_treated_as_the_persons_own_site():
    gh = "https://github.com/someone"
    html = '<html><title>x</title><body><a href="mailto:a@b.com">a</a><a href="mailto:c@d.com">c</a></body></html>'
    e = _person_enricher({gh: _Page(gh, html)})
    c = Contact(kind="person", name="Some One", urls=[gh])
    e._enrich_one(c)
    assert not c.emails


def test_snippet_address_with_the_persons_name_is_used_when_the_site_will_not_load():
    from brain.sources.websearch import Hit

    hits = [Hit("https://lab.example.edu/people", "People", "Contact: jane.doe@lab.example.edu, office 4")]
    e = _person_enricher({}, hits)
    c = Contact(kind="person", name="Jane Doe", org="Example Lab", org_domain="lab.example.edu")
    e._enrich_one(c)
    assert c.best_email and c.best_email.address == "jane.doe@lab.example.edu"


def test_organisation_with_an_unreachable_site_gets_its_address_from_a_search_snippet():
    from brain.orgs import OrgEnricher
    from brain.settings import Settings
    from brain.sources.websearch import Hit

    hits = [Hit("https://directory.example.com/x", "Royal Album, Jaipur", "Call 98xxxx or write to hello@royalalbum.in")]
    e = OrgEnricher(fetcher=_FakeFetcher({}), search=_FakeSearch(hits), settings=Settings())
    org = Contact(kind="org", name="Royal Album", org_domain="royalalbum.in")
    e.enrich(org, CampaignSpec(id="c"), [])
    assert "site_unreachable" in org.flags and org.best_email and org.best_email.address == "hello@royalalbum.in"
    other = Contact(kind="org", name="Other", org_domain="other.in")
    e2 = OrgEnricher(fetcher=_FakeFetcher({}), search=_FakeSearch([Hit("u", "t", "mail us at x@elsewhere.com")]), settings=Settings())
    e2.enrich(other, CampaignSpec(id="c"), [])
    assert other.best_email is None


def test_github_rate_limits_reach_the_warning():
    from brain.cache import Cache
    from brain.settings import Settings
    from brain.sources.websearch import WebSearch

    ws = WebSearch(Settings(), Cache())
    ws.note_provider_failure("GitHub", "HTTP 403, rate limited: set GITHUB_TOKEN", 9, 12)
    msg = ws.provider_problem()
    assert "GitHub failed 9 of 12" in msg and "GITHUB_TOKEN" in msg
    ws2 = WebSearch(Settings(), Cache())
    ws2.note_provider_failure("GitHub", "x", 1, 30)
    assert ws2.provider_problem() == ""


def test_crates_source_turns_owners_into_contacts_with_their_public_profile():
    from types import SimpleNamespace

    from brain.discovery.hooks.crates import search

    class F:
        def get_json(self, url, params=None, **kw):
            if url.endswith("/crates"):
                return {"crates": [{"id": "ratelimit", "description": "async rate limiter", "downloads": 5000, "max_version": "1.0"}]}
            if url.endswith("/owner_user"):
                return {"users": [{"login": "jdoe", "name": None, "kind": "user", "url": "https://github.com/jdoe"},
                                  {"login": "rust-lang", "kind": "user"}]}
            if "api.github.com/users/jdoe" in url:
                return {"login": "jdoe", "name": "Priya Nair", "blog": "priyanair.dev", "email": "priya@priyanair.dev",
                        "html_url": "https://github.com/jdoe", "company": "@Acme"}
            return None

    run = SimpleNamespace(deps=SimpleNamespace(fetcher=F()), spec=CampaignSpec(id="c", topics=["async rate limiting"]))
    people, stats = search(run)
    assert [p.name for p in people] == ["Priya Nair"] and stats["owners"] == 1
    p = people[0]
    assert p.best_email and p.best_email.address == "priya@priyanair.dev" and "https://priyanair.dev" in p.urls and p.org == "Acme"


def test_negations_in_the_request_become_anti_patterns():
    from brain.intake import negations_from_text

    t = "find small installers (not distributors, not manufacturers) in Kerala, excluding agencies and also big firms"
    assert negations_from_text(t) == ["distributors", "manufacturers", "agencies"]
    assert negations_from_text("professors in Germany, excluding Max Planck institutes") == []
    assert negations_from_text("find clinics in Pune") == []


def test_a_set_aside_size_limit_is_shown_as_an_assumption():
    from brain.intake import set_aside_size_note

    assert "not applied to everyone" in set_aside_size_note("small installers, and also large EPC firms with over 500 employees")
    assert set_aside_size_note("fintechs with 50 to 500 employees") == ""


def test_token_window_waits_instead_of_exceeding_the_limit():
    from brain.llm import _TokenWindow

    clock = {"t": 1000.0}
    slept = []

    def sleep(d):
        slept.append(d)
        clock["t"] += d

    w = _TokenWindow()
    assert w.acquire("h", 5000, 8000, now=lambda: clock["t"], sleep=sleep) == 0
    assert w.acquire("h", 2000, 8000, now=lambda: clock["t"], sleep=sleep) == 0
    waited = w.acquire("h", 3000, 8000, now=lambda: clock["t"], sleep=sleep)  # 7000 used: must wait for the first to age out
    assert waited > 0 and sum(slept) >= 55
    assert w.acquire("h", 10, 0, now=lambda: clock["t"], sleep=sleep) == 0  # no limit configured


def test_bench_writes_a_partial_report_after_each_scenario(tmp_path, monkeypatch):
    import brain.bench as bench

    seen = []

    def fake_run(sc, **kw):
        if seen:  # the first scenario has already been checkpointed when the second one starts
            assert (tmp_path / "t.partial.json").exists()
        seen.append(sc["name"])
        return {"name": sc["name"], "split": "dev", "target": 5, "good@K": 3, "reachable": 4, "search_calls": 1,
                "llm_paid_calls": 1, "llm_usd": 0.0}

    monkeypatch.setattr(bench, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(bench, "run_scenario", fake_run)
    rep = bench.run_bench(["academic_gnn", "b2b_dental_bangalore"], label="t", progress=lambda m: None)
    assert len(rep["rows"]) == 2 and not (tmp_path / "t.partial.json").exists() and (tmp_path / "t.json").exists()
