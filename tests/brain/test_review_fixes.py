"""Fixes from docs/improvement-review.md: whole-word region match, Cloudflare email decoding, particle surnames,
send-time suppression in the bridge, pre-judge/judge agreement."""

from brain.draft import salutation, surname
from brain.emailfinder import decode_cfemail
from brain.models import CampaignSpec, Contact, Draft, EmailCandidate, SenderProfile, TargetPersona
from brain.pages import parse
from brain.pipeline import note_prejudge_agreement
from brain.playbooks import get_playbook
from brain.rank import filter_contacts
from brain.store import Store


def _cf(email: str, key: int = 0x4A) -> str:
    return bytes([key] + [b ^ key for b in email.encode()]).hex()


def test_region_match_is_whole_word():
    spec = CampaignSpec(id="r", pack="speakers", objective="Speakers", persona=TargetPersona(description="x"),
                        topics=["ai"], regions=["Goa"], target_count=3, sender=SenderProfile(name="T"))
    pb = get_playbook("speakers")
    goa = Contact(name="Asha Rao", location_text="Panjim, Goa, India", bio="ai")
    goals = Contact(name="Ravi Iyer", location_text="Chennai", org="Goals Academy", bio="ai")
    kept, dropped = filter_contacts([goa, goals], spec, pb)
    assert [c.name for c in kept] == ["Asha Rao"] and dropped == {"region": 1}


def test_cloudflare_protected_email_is_decoded_from_page():
    assert decode_cfemail(_cf("asha.rao@iitd.ac.in")) == "asha.rao@iitd.ac.in"
    assert decode_cfemail("zz") == "" and decode_cfemail("") == ""
    html = (f'<html><body><h1>Asha Rao</h1><p>Contact: <a class="__cf_email__" data-cfemail="{_cf("asha.rao@iitd.ac.in")}">'
            f'[email protected]</a></p><a href="/cdn-cgi/l/email-protection#{_cf("dept@iitd.ac.in")}">mail</a></body></html>')
    page = parse(html, "https://iitd.ac.in/asha")
    assert "asha.rao@iitd.ac.in" in page.emails and "dept@iitd.ac.in" in page.mailtos
    assert not any("email-protection" in u for u, _ in page.links + page.nav_links)


def test_surname_keeps_particles_and_drops_suffixes():
    assert surname("Anna van der Berg".split()) == "van der Berg"
    assert surname(["Rao"]) == "Rao" and surname("Priya Nair".split()) == "Nair"
    assert surname("John Smith Jr.".split()) == "Smith"
    assert salutation(Contact(name="Anna van der Berg"), get_playbook("academic")) == "Dear Professor van der Berg,"


def test_bridge_skips_suppressed_recipients(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("langgraph")  # the legacy review stack (orchestrator.state) is not in requirements-brain.txt
    saved = []
    from db import database as legacy  # the real module, with its writes replaced so no legacy DB is touched

    monkeypatch.setattr(legacy, "init_db", lambda: None)
    monkeypatch.setattr(legacy, "upsert_session_state", lambda *a, **k: None)
    monkeypatch.setattr(legacy, "save_professor", lambda p, session_id=None: saved.append(p.email))
    monkeypatch.setattr(legacy, "save_draft_email", lambda *a, **k: None)
    from brain.bridge import export_to_review

    store = Store(f"sqlite:///{tmp_path / 'b.db'}")
    spec = CampaignSpec(id="b", pack="general", objective="o", persona=TargetPersona(description="x"),
                        topics=["t"], target_count=2, sender=SenderProfile(name="T"))
    store.save_campaign(spec, "done", {})
    people = []
    for n, addr in (("Asha Rao", "asha@good.org"), ("Ravi Iyer", "ravi@blocked.org")):
        c = Contact(id=n.split()[0].lower(), name=n, org_domain=addr.split("@")[1], emails=[EmailCandidate(address=addr, method="text", confidence=0.9)])
        people.append(c)
    drafts = {c.id: Draft(contact_id=c.id, to=c.emails[0].address, subject="Hello there", body="Hi, a short note.")
              for c in people}
    store.upsert_contacts(people)
    store.save_results("b", people, drafts)
    store.suppress("blocked.org", "asked to stop")  # added after discovery
    assert export_to_review(store, spec) == 1 and saved == ["asha@good.org"]


def test_prejudge_agreement_counts_disagreements():
    funnel = {}
    a = Contact(name="A", prejudge=0.3); a.score = 0.8       # screen too strict
    b = Contact(name="B", prejudge=0.8); b.score = 0.4       # screen too lenient
    c = Contact(name="C", prejudge=0.8); c.score = 0.9       # agree
    d = Contact(name="D"); d.score = 0.9                     # never screened: ignored
    note_prejudge_agreement(funnel, [a, b, c, d])
    assert funnel["prejudge_agreement"] == {"both": 3, "low_pre_good": 1, "high_pre_bad": 1}
