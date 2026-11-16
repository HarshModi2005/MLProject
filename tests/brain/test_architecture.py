"""Uploads, campaign context, search planning, tools and playbooks — offline."""

import json

import pytest

from brain.context import CampaignContext
from brain.models import CampaignSpec, SenderProfile, TargetPersona
from brain.playbooks import PLAYBOOKS, catalog_text, get_playbook
from brain.searchplan import SearchPlanner, search_budget
from brain.tools import TOOLS, authors_from_article
from brain.uploads import apply, classify, load, rows_to_contacts


def _spec(**kw):
    base = dict(id="a", pack="sponsorship", objective="Sponsor our fest", topics=["developer tools", "edtech"],
                expanded_terms=["coding platforms"], countries=["IN"], regions=["Bangalore"],
                persona=TargetPersona(description="brands"), target_count=5, sender=SenderProfile(name="A"))
    base.update(kw)
    return CampaignSpec(**base)


# ── uploads ──────────────────────────────────────────────────────────────────
def test_upload_roles_are_inferred_and_applied(tmp_path):
    (tmp_path / "sponsors_2026.csv").write_text("Company,Website\nDevfolio,devfolio.co\nGeeksforGeeks,geeksforgeeks.org\n")
    (tmp_path / "outreach_log.csv").write_text("Name,Email,Status\nAsha Rao,asha@acme.com,replied\n")
    (tmp_path / "dnc.txt").write_text("spam@corp.com\nbadbrand.com\n")
    (tmp_path / "leads.csv").write_text("Name,Company,Email,Title\nDana Lee,Acme,dana@acme.io,CTO\n,Smile Dental,info@smile.in,\n")
    (tmp_path / "old_email.txt").write_text("Hi Priya,\n\nWe run Advitiya, a fest with 5000 students. "
                                            "Would you like to sponsor our hackathon this year? It reaches every "
                                            "engineering college in the region and we can share detailed numbers.\n\n"
                                            "Thanks,\nHarsh\n")
    (tmp_path / "deck.pdf").write_bytes(b"")  # unreadable files must not break classification of the rest
    ctx = CampaignContext()
    roles = {}
    targets = []
    for name in ["sponsors_2026.csv", "outreach_log.csv", "dnc.txt", "leads.csv", "old_email.txt"]:
        u = classify(load(str(tmp_path / name)))
        roles[name] = u.role
        targets += apply(u, ctx)
    assert roles == {"sponsors_2026.csv": "examples", "outreach_log.csv": "exclude", "dnc.txt": "exclude",
                     "leads.csv": "targets", "old_email.txt": "style"}
    assert ctx.seed_names == ["Devfolio", "GeeksforGeeks"] and "devfolio.co" in ctx.seed_domains
    assert "asha@acme.com" in ctx.exclude_emails and "spam@corp.com" in ctx.exclude_emails
    assert "badbrand.com" in ctx.exclude_domains
    assert ctx.style_examples and "Advitiya" in ctx.style_examples[0]
    by_name = {c.name: c for c in targets}
    assert by_name["Dana Lee"].kind == "person" and by_name["Dana Lee"].best_email.address == "dana@acme.io"
    assert by_name["Smile Dental"].kind == "org" and by_name["Smile Dental"].best_email.method == "inbox"


def test_excel_upload(tmp_path):
    import pandas as pd

    pd.DataFrame({"Organisation": ["Acme"], "E-mail": ["hello@acme.com"]}).to_excel(tmp_path / "list.xlsx", index=False)
    u = classify(load(str(tmp_path / "list.xlsx")))
    assert u.role == "targets" and rows_to_contacts(u.rows, "x")[0].best_email.address == "hello@acme.com"


def test_context_exclusions_and_sourced_facts():
    ctx = CampaignContext()
    ctx.add_fact("5000+ attendees in 2026", "user")
    ctx.add_fact("unsourced claim", "")  # dropped: every fact needs a source
    assert [f.text for f in ctx.facts] == ["5000+ attendees in 2026"]
    ctx.add_fact("Advitiya 2026: 5000+ attendees in 2026 overall", "cv")  # same claim, reworded
    ctx.add_fact("6000+ attendees in 2027", "user")  # different numbers: a different fact
    assert [f.text for f in ctx.facts] == ["5000+ attendees in 2026", "6000+ attendees in 2027"]
    for v in ["Bad Brand", "x@y.com", "https://www.evil.co.in/page"]:
        ctx.exclude(v)
    assert ctx.is_excluded(name="bad brand") and ctx.is_excluded(emails=["X@Y.com"])
    assert ctx.is_excluded(domain="shop.evil.co.in") and not ctx.is_excluded(name="Good Brand")


# ── search planner ───────────────────────────────────────────────────────────
def test_planner_stops_early_and_never_repeats_queries():
    ctx = CampaignContext()
    spec = _spec(search_queries=["devtools sponsors India", "edtech sponsors India", "devtools sponsors india"],
                 lookalike_queries=["companies like Devfolio"])
    planner = SearchPlanner(spec, get_playbook("sponsorship"), ctx, llm=None, need=15)
    found = []

    def execute(q):
        found.extend(range(10))  # every query yields 10 new candidates
        return 10, 10

    planner.run(execute, have=lambda: len(found), budget_left=lambda: 99)
    # Lookalikes first; stops after 2 queries (20 >= 15); the near-duplicate query is never sent.
    assert [q.query for q in ctx.query_log] == ["companies like Devfolio", "devtools sponsors India"]


def test_planner_expands_only_from_productive_queries_and_respects_budget():
    ctx = CampaignContext()
    spec = _spec(search_queries=["developer tools sponsors Bangalore", "random query"])
    planner = SearchPlanner(spec, get_playbook("sponsorship"), ctx, llm=None, need=1000)
    calls = []

    def execute(q):
        calls.append(q)
        return (10, 5) if "developer tools" in q.lower() or "edtech" in q.lower() else (10, 0)

    planner.run(execute, have=lambda: 0, budget_left=lambda: 12 - len(calls))
    assert len(calls) <= 12
    variants = [q for q in calls if q not in planner.initial()]
    # Round 2 rewrote the productive query (other topic / other place), never the unproductive one.
    assert variants and all("random" not in v for v in variants)
    assert any("edtech" in v.lower() or "coding platforms" in v.lower() for v in variants)
    assert search_budget(5, 80) == 25 and search_budget(50, 80) == 80


# ── tools and playbooks ──────────────────────────────────────────────────────
def test_registry_covers_every_playbook_source():
    for pb in PLAYBOOKS.values():
        for source in pb.sources:
            assert source in TOOLS, (pb.name, source)
    assert "sponsorship" in catalog_text() and "b2b_sales" in catalog_text()


def test_user_playbook_from_json(tmp_path, monkeypatch):
    import importlib

    import brain.playbooks as pbmod

    (tmp_path / "alumni.json").write_text(json.dumps({
        "name": "alumni", "title": "Alumni outreach", "description": "Alumni of a university for mentoring",
        "target_kind": "person", "sources": ["web"], "unknown_field": 1}))
    monkeypatch.setenv("BRAIN_PLAYBOOKS_DIR", str(tmp_path))
    importlib.reload(pbmod)
    try:
        assert pbmod.get_playbook("alumni").title == "Alumni outreach"
    finally:
        monkeypatch.delenv("BRAIN_PLAYBOOKS_DIR")
        importlib.reload(pbmod)


def test_bylines_from_article_pages():
    ld = {"@type": "NewsArticle", "author": [{"@type": "Person", "name": "Ananya Rao", "url": "https://x.com/ananya"}]}
    html = f'<html><head><script type="application/ld+json">{json.dumps(ld)}</script>' \
           '<meta name="author" content="Ananya Rao, Vikram Shah"></head><body>text</body></html>'
    assert authors_from_article(html, "https://news.example/a") == [("Ananya Rao", "https://x.com/ananya"),
                                                                    ("Vikram Shah", None)]
    assert authors_from_article("<html><body><p>By Meera Iyer | Updated</p></body></html>", "u") == [("Meera Iyer", None)]
    assert authors_from_article('<meta name="author" content="Staff Reporter">', "u") == []


def test_places_tool_keeps_businesses_with_websites(tmp_path):
    from brain.cache import Cache
    from brain.orgs import discover_places
    from brain.settings import Settings
    from brain.sources.websearch import Place, WebSearch

    class FakeMaps(WebSearch):
        def places(self, query, gl=""):
            self.calls += 1
            return [Place("Smile Dental", "https://smiledental.in/", "12 CMH Rd, Bengaluru", category="Dentist",
                          rating=4.8, reviews=210),
                    Place("No Site Clinic", "", "Bengaluru"),
                    Place("Practo listing", "https://www.practo.com/x", "Bengaluru")]

    st = Settings()
    st.serper_api_key = "x"
    maps = FakeMaps(st, Cache())
    ctx = CampaignContext()
    spec = _spec(pack="b2b_sales", topics=["dental clinic"])
    orgs, stats = discover_places(spec, get_playbook("b2b_sales"), maps, SearchPlanner(spec, get_playbook("b2b_sales"), ctx))
    assert [o.name for o in orgs] == ["Smile Dental"]
    assert orgs[0].location_text == "12 CMH Rd, Bengaluru" and orgs[0].org_domain == "smiledental.in"
    assert ctx.query_log[0].query == "dental clinic in Bangalore" and ctx.query_log[0].kind == "places"
