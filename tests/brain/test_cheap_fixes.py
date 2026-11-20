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
