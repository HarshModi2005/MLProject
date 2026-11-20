"""Address safety: an email must reach the person it was written for (production-hardening, item 1)."""

import smtplib

from brain.discovery.reach import verify
from brain.discovery.reach.tiers import contact_tier, counts, email_tier, is_sendable
from brain.emailfinder import name_match, score_email
from brain.enrich import has_usable_email
from brain.models import Contact, Draft, EmailCandidate


def E(addr, method="text", conf=0.8, **kw):
    return EmailCandidate(address=addr, confidence=conf, method=method, **kw)


# ── identity: whose address is it? ───────────────────────────────────────────
def test_first_name_with_another_surname_is_someone_else():
    assert name_match("Rahul Sharma", "rahul.gupta@x.com") == 0  # was 1.0: any Rahul matched
    assert name_match("Shampa Raghunathan", "shampa.santra@gmail.com") == 0
    assert name_match("Priya Nair", "priya.menon@x.com") == 0


def test_legitimate_name_forms_still_match():
    for name, addr in [("Rahul Sharma", "rahul@x.com"), ("Rahul Sharma", "rahul.sharma@x.com"),
                       ("Rahul Kumar Sharma", "rahul.kumar.sharma@x.com"), ("Rahul Sharma", "drrahulsharma@x.com"),
                       ("Rahul Sharma", "sharmar@x.com"), ("Rahul Sharma", "rahul.cs@x.com"),
                       ("Rahul Sharma", "rsharma@x.com"), ("Rahul Sharma", "rahul_sharma@x.com")]:
        assert name_match(name, addr) == 1.0, (name, addr)


def test_an_address_at_another_organisation_does_not_clear_the_send_threshold():
    msr = Contact(name="Rahul Sharma", org="Microsoft Research India")  # org known by name only
    assert score_email("sharmar@cs.stanford.edu", msr, "text") < 0.6
    assert score_email("sharmar@cs.stanford.edu", msr, "mailto") < 0.6
    corp = Contact(name="Rahul Sharma", org="Microsoft", org_domain="microsoft.com")
    assert score_email("rahul.gupta@microsoft.com", corp, "text") < 0.6  # a colleague, not him
    assert score_email("rahul.sharma@microsoft.com", corp, "text") >= 0.9
    # ...but the person's own page listing a second affiliation is still believed
    assert score_email("sharmar@cs.stanford.edu", msr, "mailto", on_own_page=True) >= 0.6


def test_orcid_address_with_a_different_surname_is_not_trusted_blindly():
    c = Contact(name="Shampa Raghunathan")
    assert score_email("shampa.santra@gmail.com", c, "orcid", on_own_page=True) < 0.6
    assert score_email("shampa.r@univ.edu", Contact(name="Shampa Raghunathan", org_domain="univ.edu"), "orcid", True) >= 0.6


def test_unattributed_address_is_never_reachable():
    c = Contact(name="Rahul Sharma", org_domain="x.com", emails=[E("rahul.gupta@x.com", "mailto", 0.9)])
    assert contact_tier(c) == "unattributed" and not has_usable_email(c, 0.6)
    ok = Contact(name="Rahul Sharma", org_domain="x.com", emails=[E("rahul.sharma@x.com", "mailto", 0.9)])
    assert has_usable_email(ok, 0.6)


# ── tiers ────────────────────────────────────────────────────────────────────
def test_tiers():
    p = Contact(name="Anita Rao", org_domain="x.com")
    assert email_tier(p, E("anita.rao@x.com", "mailto")) == "attributed"
    assert email_tier(p, E("anita.rao@x.com", "mailto", deliverability="deliverable", verified_by="smtp")) == "verified"
    assert email_tier(p, E("anita.rao@x.com", "pattern_global", 0.55)) == "guess"
    assert email_tier(p, E("anita.rao@x.com", "pattern_global", 0.55, deliverability="catch_all")) == "guess"
    assert email_tier(p, E("anita.rao@x.com", "pattern_global", 0.55, deliverability="deliverable")) == "verified"
    assert email_tier(p, E("bob.jones@x.com", "text")) == "unattributed"
    assert email_tier(p, E("anita.rao@x.com", "orcid", deliverability="undeliverable")) == "unattributed"
    assert email_tier(p, None) == "none"
    org = Contact(kind="org", name="Voxa", org_domain="voxa.ai")
    assert email_tier(org, E("partnerships@voxa.ai", "inbox", 0.7)) == "inbox"
    assert email_tier(org, E("partnerships@voxa.ai", "inbox", 0.7, deliverability="deliverable")) == "verified"
    assert email_tier(Contact(name="A B"), E("whoever@y.org", "provided", 0.95)) == "attributed"  # the user's own list
    assert counts([p, org]) == {"none": 2}


def test_only_a_clean_draft_to_a_verified_address_is_sendable():
    assert Draft(contact_id="1", to="a@b.c", subject="s", body="b", address_tier="verified").sendable
    assert not Draft(contact_id="1", to="a@b.c", subject="s", body="b", address_tier="attributed").sendable
    assert not Draft(contact_id="1", to="a@b.c", subject="s", body="b", address_tier="verified", issues=["x"]).sendable
    assert is_sendable(Contact(name="Anita Rao", emails=[E("anita.rao@x.com", "mailto", deliverability="deliverable")]))


# ── mailbox verification ─────────────────────────────────────────────────────
class FakeSMTP:
    """Scripted mail server: `book` maps local part -> (code, text); anything else -> default."""
    book: dict = {}
    default = (550, "5.1.1 user unknown")
    log: list = []
    boom: Exception = None

    def __init__(self, timeout=None):
        pass

    def connect(self, host, port):
        if FakeSMTP.boom:
            raise FakeSMTP.boom

    def ehlo(self, name):
        return 250, b"ok"

    def mail(self, sender):
        return 250, b"ok"

    def rcpt(self, addr):
        FakeSMTP.log.append(addr)
        code, text = FakeSMTP.book.get(addr.split("@")[0], FakeSMTP.default)
        return code, text.encode()

    def quit(self):
        pass


def smtp(**kw):
    FakeSMTP.book, FakeSMTP.default, FakeSMTP.log, FakeSMTP.boom = {}, (550, "5.1.1 user unknown"), [], None
    return verify.SMTPVerifier(factory=FakeSMTP, mx_lookup=lambda d: ["mx." + d], **kw)


def test_smtp_probe_verdicts():
    v = smtp()
    FakeSMTP.book = {"anita.rao": (250, "ok")}
    assert v.verdict("anita.rao@x.com") == "deliverable"
    assert v.verdict("nobody@x.com") == "undeliverable"  # 550 user unknown, domain is not catch-all
    FakeSMTP.book["greylisted"] = (451, "try later")
    assert v.verdict("greylisted@x.com") == "unknown"
    FakeSMTP.book["blocked"] = (554, "5.7.1 rejected: blocked by spam policy")
    assert v.verdict("blocked@x.com") == "unknown"  # a policy refusal is not "no such mailbox"


def test_catch_all_domains_prove_nothing():
    v = smtp()
    FakeSMTP.default = (250, "ok")  # accepts every recipient, including the random probe
    assert v.verdict("anyone@catchall.org") == "catch_all"


def test_smtp_never_claims_what_it_cannot_know():
    v = smtp()
    assert v.verdict("someone@gmail.com") == "unknown"  # opaque provider
    FakeSMTP.boom = OSError("port 25 blocked")
    assert v.verdict("someone@x.com") == "unknown"
    assert smtp(max_checks=0).verdict("a@b.com") == "unknown"
    assert verify.SMTPVerifier(factory=FakeSMTP, mx_lookup=lambda d: []).verdict("a@b.com") == "unknown"  # no MX


def test_verify_addresses_stamps_verdicts_and_falls_back_to_the_runner_up():
    v = smtp()
    FakeSMTP.book = {"anita.rao": (250, "ok")}
    c = Contact(name="Anita Rao", org_domain="x.com", emails=[E("a.rao@x.com", "pattern_global", 0.62),
                                                              E("anita.rao@x.com", "mailto", 0.55)])
    stats = verify.verify_addresses([c], vs=[v])
    assert [e.address for e in c.emails] == ["anita.rao@x.com"]  # the dead guess was dropped
    assert c.best_email.deliverability == "deliverable" and c.best_email.verified_by == "smtp"
    assert stats["undeliverable"] == 1 and stats["deliverable"] == 1
    assert contact_tier(c) == "verified"


def test_verification_is_not_repeated_and_mx_alone_never_verifies(monkeypatch):
    monkeypatch.setattr(verify, "has_mx", lambda d: True)
    c = Contact(name="Anita Rao", emails=[E("anita.rao@x.com", "mailto")])
    assert verify.verify_addresses([c], vs=[verify.MXVerifier()])["unknown"] == 1
    assert contact_tier(c) == "attributed"  # a domain that accepts mail is not a mailbox that exists
    assert verify.verify_addresses([c], vs=[verify.MXVerifier()])["checked"] == 0  # already has a verdict


def test_hunter_statuses(monkeypatch):
    class F:
        paid_calls = {}

        def __init__(self, status):
            self.status = status

        def get_json(self, url, params=None, headers=None, **kw):
            return {"data": {"status": self.status}}

    for status, want in [("valid", "deliverable"), ("invalid", "undeliverable"), ("accept_all", "catch_all"),
                         ("webmail", "unknown"), ("unknown", "unknown")]:
        assert verify.HunterVerifier(F(status), "k").verdict("a@b.com") == want


def test_smtp_gives_up_quickly_when_the_network_blocks_mail_servers():
    v = smtp()
    FakeSMTP.boom = OSError("connection timed out")
    for i in range(5):
        assert v.verdict(f"p{i}@dead-net-{i}.com") == "unknown"
    assert v.disabled and len(FakeSMTP.log) == 0  # no more connections were attempted after the third failure
    FakeSMTP.boom = None
    FakeSMTP.book = {"anita": (250, "ok")}
    assert v.verdict("anita@x.com") == "unknown"  # stays off for the rest of the run


def test_smtp_needs_a_real_sender_address_to_be_enabled(monkeypatch):
    monkeypatch.setenv("BRAIN_SMTP_VERIFY", "1")
    monkeypatch.delenv("BRAIN_SMTP_FROM", raising=False)
    monkeypatch.delenv("HUNTER_API_KEY", raising=False)
    assert [v.name for v in verify.verifiers()] == ["mx"]
    monkeypatch.setenv("BRAIN_SMTP_FROM", "checks@mydomain.com")
    assert [v.name for v in verify.verifiers()] == ["mx", "smtp"]
