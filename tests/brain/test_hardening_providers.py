"""Paid contact data as the default route for named people inside companies (production-hardening, item 3)."""

from brain.discovery.reach import providers
from brain.discovery.reach.tiers import contact_tier
from brain.models import Contact


class F:
    def __init__(self, payload):
        self.payload, self.calls, self.paid_calls = payload, [], {}

    def get_json(self, url, params=None, headers=None, **kw):
        self.calls.append((url, params))
        return self.payload


def org():
    return Contact(kind="org", name="Voxa", org_domain="voxa.ai", sources=["web"])


def people(*rows):
    return {"data": {"emails": [dict({"type": "personal", "confidence": 90}, **r) for r in rows]}}


def test_role_match_beats_a_more_confident_junior(monkeypatch):
    monkeypatch.setenv("HUNTER_API_KEY", "k")
    f = F(people({"value": "dev@voxa.ai", "first_name": "Dev", "last_name": "Rao", "position": "Software Engineer", "confidence": 99},
                 {"value": "mira@voxa.ai", "first_name": "Mira", "last_name": "Iyer", "position": "Head of Partnerships", "confidence": 70}))
    c = providers.hunter_person(f, org(), ["Head of Partnerships"])
    assert c.name == "Mira Iyer" and c.best_email.address == "mira@voxa.ai" and "role_approx" not in c.flags


def test_a_senior_person_is_flagged_and_a_junior_is_never_returned(monkeypatch):
    monkeypatch.setenv("HUNTER_API_KEY", "k")
    senior = F(people({"value": "sam@voxa.ai", "first_name": "Sam", "last_name": "Rao", "position": "Co-Founder"}))
    c = providers.hunter_person(senior, org(), ["Head of Partnerships"])
    assert c.name == "Sam Rao" and "role_approx" in c.flags  # the reviewer is told this is not the exact role
    junior = F(people({"value": "dev@voxa.ai", "first_name": "Dev", "last_name": "Rao", "position": "Software Engineer"}))
    assert providers.hunter_person(junior, org(), ["Head of Partnerships"]) is None


def test_hunters_own_mailbox_check_becomes_a_verified_address(monkeypatch):
    monkeypatch.setenv("HUNTER_API_KEY", "k")
    f = F(people({"value": "mira.iyer@voxa.ai", "first_name": "Mira", "last_name": "Iyer", "position": "Head of Partnerships",
                  "verification": {"status": "valid"}}))
    c = providers.hunter_person(f, org(), ["Head of Partnerships"])
    assert c.best_email.deliverability == "deliverable" and c.best_email.verified_by == "hunter"
    assert contact_tier(c) == "verified"
    bad = F(people({"value": "x@voxa.ai", "first_name": "X", "last_name": "Y", "position": "CEO", "verification": {"status": "invalid"}}))
    assert providers.hunter_person(bad, org(), ["CEO"]) is None  # a dead mailbox is not offered


def test_no_key_no_call_and_the_cap_is_respected(monkeypatch):
    monkeypatch.delenv("HUNTER_API_KEY", raising=False)
    f = F(people({"value": "a@voxa.ai", "first_name": "A", "last_name": "B", "position": "CEO"}))
    assert providers.hunter_person(f, org(), ["CEO"]) is None and f.calls == []
    monkeypatch.setenv("HUNTER_API_KEY", "k")
    assert providers.hunter_person(f, org(), ["CEO"], cap=1) is not None
    assert providers.hunter_person(f, org(), ["CEO"], cap=1) is None  # per-campaign ceiling


def test_email_finder_for_a_named_decision_maker(monkeypatch):
    monkeypatch.setenv("HUNTER_API_KEY", "k")
    f = F({"data": {"email": "Mira.Iyer@voxa.ai", "score": 91, "verification": {"status": "valid"}}})
    e = providers.hunter_find(f, "Mira Iyer", "Head of Partnerships", org())
    assert e.address == "mira.iyer@voxa.ai" and e.method == "hunter" and e.deliverability == "deliverable"
    assert f.calls[0][1] == {"domain": "voxa.ai", "first_name": "Mira", "last_name": "Iyer"}
    assert providers.hunter_find(F({"data": {"email": "m@voxa.ai", "score": 20}}), "Mira Iyer", None, org()) is None  # weak
    assert providers.hunter_find(f, "Cher", None, org()) is None  # needs a first and a last name


def test_decision_maker_step_prefers_the_provider_and_pays_once(monkeypatch):
    from brain.orgs import OrgEnricher
    from brain.settings import Settings

    monkeypatch.setenv("HUNTER_API_KEY", "k")
    f = F(people({"value": "mira@voxa.ai", "first_name": "Mira", "last_name": "Iyer", "position": "Head of Partnerships"}))

    class NoSearch:
        remaining = 50
        available = True

        def search(self, *a, **k):
            raise AssertionError("a web search was spent although the provider had the answer")

    e = OrgEnricher(f, NoSearch(), Settings(), find_people=True)
    o = org()
    dm = e._find_decision_maker(o, ["Head of Partnerships"], [])
    assert dm.name == "Mira Iyer" and o.decision_maker["name"] == "Mira Iyer" and "hunter_tried" in o.flags
    assert e.stats["hunter_contacts"] == 1 and len(f.calls) == 1
    # a second look at the same organisation goes nowhere near the provider again
    assert "hunter_tried" in o.flags


def test_a_missing_provider_is_reported_for_person_at_company_campaigns():
    from brain.pipeline import PERSON_AT_COMPANY_FAMILIES
    from brain.playbooks import get_playbook

    for name in ("sponsorship", "job_search", "pr_media", "fundraising", "b2b_sales"):
        assert get_playbook(name).family in PERSON_AT_COMPANY_FAMILIES, name
    assert get_playbook("academic").family not in PERSON_AT_COMPANY_FAMILIES  # OpenAlex/ORCID work without one
