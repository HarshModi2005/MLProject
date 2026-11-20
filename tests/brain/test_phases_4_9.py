"""Discovery v3 phases 4-9: entity graph and walks, email conventions and verification, persistent index and
refresh, warm paths, new playbooks and Hacker News, paid plugins. No network."""

import json

import pytest

from brain.discovery import graph
from brain.discovery.reach import patterns
from brain.models import Contact, EmailCandidate, Evidence
from brain.playbooks import PLAYBOOKS, get_playbook
from brain.store import Store
from brain.tools import TOOLS
from tests.brain.test_discovery_v3 import Deps, FakeFetcher, FakeSearch, _run, _spec


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("brain.discovery.policy.time.sleep", lambda s: None)


def _sponsor(name, dom, event_url, event="HackFest 2025"):
    return Contact(kind="org", name=name, org=name, org_domain=dom, id=dom, sources=["events"],
                   evidence=[Evidence(kind="page", text=f"Sponsor or partner of {event}", url=event_url)],
                   signals={"past_sponsorship": 1.0})


# ── phase 4: graph ───────────────────────────────────────────────────────────
def test_graph_edges_come_from_sourced_evidence():
    s = _sponsor("DevToolCo", "devtool.io", "https://hackfest.in/2025/sponsors")
    nodes, edges = graph.extract(s)
    assert edges[0]["type"] == "sponsored" and edges[0]["dst"] == "event:hackfest.in/2025"
    j = Contact(kind="person", name="Priya Nair", title="Journalist", org="The Paper", org_domain="paper.in")
    assert graph.extract(j)[1][0] == {"src": "person:priya nair|paper.in", "dst": "outlet:paper.in", "type": "wrote_for",
                                      "source": "?", "url": None}


def test_graph_walk_finds_co_sponsors_from_earlier_campaigns(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'g.db'}")
    a = _sponsor("DevToolCo", "devtool.io", "https://hackfest.in/2025/sponsors")
    b = _sponsor("CloudCo", "cloud.co", "https://hackfest.in/2025/partners")
    c = _sponsor("Other", "other.io", "https://elsewhere.org/2024/sponsors", "Other Fest")
    b.add_email(EmailCandidate(address="partners@cloud.co", confidence=0.7, method="inbox"))
    store.upsert_contacts([a, b, c])
    assert graph.record(store, [a, b, c]) == 3
    near = graph.neighbours(store, [a])
    assert [n["node"] for n in near] == ["org:cloud.co"] and near[0]["via"] == "HackFest 2025"
    deps = Deps()
    deps.store = store
    found, stats = TOOLS["graph_walk"].fn(_run(_spec(target_kind="org"), deps, pb="sponsorship", winners=[a]))
    assert [x.name for x in found] == ["CloudCo"] and stats["from_memory"] == 1
    assert found[0].best_email.address == "partners@cloud.co" and "Also sponsored HackFest 2025" in found[0].evidence[-1].text


def test_sponsor_walk_searches_from_the_winning_sponsors(monkeypatch):
    seen = {}

    def fake_events(run, queries=None, source="events"):
        seen["q"], seen["source"] = queries, source
        return [], {}

    monkeypatch.setattr("brain.channels.discover_events", fake_events)
    w = _sponsor("DevToolCo", "devtool.io", "https://hackfest.in/2025/sponsors")
    TOOLS["sponsor_walk"].fn(_run(_spec(target_kind="org", regions=["Bangalore"]), Deps(), pb="sponsorship", winners=[w]))
    assert seen["q"][0] == '"DevToolCo" sponsors hackathon Bangalore' and seen["source"] == "sponsor_walk"


def test_beat_walk_finds_colleagues_on_the_winners_outlet():
    from brain.sources.websearch import NewsHit

    art = '<html><head><meta name="author" content="Rohan Das"></head></html>'
    news = {"site:paper.in speech recognition": [NewsHit("https://paper.in/x", "Speech AI", "s", "The Paper", "1d")]}
    deps = Deps(FakeFetcher(pages={"https://paper.in/x": art}), FakeSearch(news=news))
    w = Contact(kind="person", name="Priya Nair", title="Journalist", org_domain="paper.in")
    found, stats = TOOLS["beat_walk"].fn(_run(_spec(), deps, winners=[w]))
    assert [c.name for c in found] == ["Rohan Das"] and found[0].sources == ["beat_walk"]


# ── phase 5: email conventions ───────────────────────────────────────────────
def test_conventions_learned_in_one_campaign_serve_the_next(tmp_path, monkeypatch):
    store = Store(f"sqlite:///{tmp_path / 'p.db'}")
    found = [Contact(kind="person", name=n, emails=[EmailCandidate(address=a, confidence=0.85, method="mailto")])
             for n, a in [("Asha Rao", "asha.rao@paper.in"), ("Vikram Shah", "vikram.shah@paper.in"),
                          ("Lena Ortiz", "lena@gmail.com")]]
    assert patterns.votes(found) == {"paper.in": {"first.last": 2}}
    patterns.learn(store, found)
    assert store.domain_pattern("paper.in") == ("first.last", 2)
    monkeypatch.setattr(patterns, "has_mx", lambda d: True)
    new = Contact(kind="person", name="Meera Pillai", org_domain="paper.in")
    g = patterns.guess(new, store)
    assert g.address == "meera.pillai@paper.in" and g.method == "pattern_global" and g.confidence == 0.55
    startup = Contact(kind="person", name="Kiran Rao", org="Voxa", org_domain="voxa.ai")
    assert patterns.guess(startup, store) is None  # no convention known, and guesses not accepted
    assert patterns.guess(startup, store, accept_prior=True).address == "kiran@voxa.ai"
    monkeypatch.setattr(patterns, "has_mx", lambda d: None)  # cannot check the domain: no prior guess
    assert patterns.guess(startup, store, accept_prior=True) is None
    assert patterns.threshold(type("S", (), {"min_email_confidence": 0.6})(), get_playbook("job_search")) == 0.5
    assert patterns.threshold(type("S", (), {"min_email_confidence": 0.6})(), get_playbook("academic")) == 0.6


def test_undeliverable_guesses_are_removed(monkeypatch):
    from brain.discovery.reach import verify

    monkeypatch.setattr(verify, "has_mx", lambda d: d != "dead.io")
    a = Contact(name="A B", emails=[EmailCandidate(address="a@dead.io", confidence=0.5, method="pattern_prior")],
                flags=["email_guessed"])
    b = Contact(name="C D", emails=[EmailCandidate(address="c@live.io", confidence=0.5, method="pattern_prior")])
    stats = verify.verify_guesses([a, b])
    assert stats == {"checked": 2, "removed": 1, "confirmed": 0} and a.emails == [] and "email_guessed" not in a.flags
    assert b.best_email.address == "c@live.io"


# ── phase 6: index ───────────────────────────────────────────────────────────
def test_remembered_proof_expires_and_refresh_renews_it(tmp_path):
    from brain.discovery.index import decay, refresh

    c = Contact(kind="org", name="Voxa", signals={"hiring": 1.0, "past_sponsorship": 1.0})
    assert decay(c, 40) == ["hiring"] and "past_sponsorship" in c.signals and "stale:hiring" in c.flags
    store = Store(f"sqlite:///{tmp_path / 'r.db'}")
    org = Contact(kind="org", name="Voxa", org_domain="voxa.ai", id="voxa", keys=["ats:workable:voxa"], signals={"hiring": 1.0},
                  emails=[EmailCandidate(address="hello@voxa.ai", confidence=0.7, method="inbox")])
    store.upsert_contacts([org])
    f = FakeFetcher(api={"https://apply.workable.com/api/v1/widget/accounts/voxa": {"name": "Voxa", "jobs": [
        {"title": "ML Engineer", "city": "Bengaluru", "country": "India", "url": "u", "published_on": "2026-11-01"}]}})
    f.get_json = lambda url, params=None, ttl=0, headers=None: f.api.get(url)
    stats = refresh(store, f)
    assert stats["updated"] == 1 and stats["still_hiring"] == 1
    again = store.contact("voxa")
    assert again.signals["hiring"] == 1.0 and again.evidence[0].text.startswith("Hiring now: ML Engineer")


# ── phase 7: warm paths ──────────────────────────────────────────────────────
LINKEDIN = """Notes:
"When exporting your connection data, you may notice that some of the email addresses are missing."

First Name,Last Name,URL,Email Address,Company,Position,Connected On
Priya,Nair,https://www.linkedin.com/in/x,priya@voxa.ai,Voxa AI,Head of Engineering,12 Mar 2024
Arjun,Mehta,https://www.linkedin.com/in/y,,Paper Media Pvt Ltd,Editor,01 Jan 2023
"""


def test_warm_paths_from_the_users_own_exports(tmp_path):
    from brain.discovery.warm.importers import read_csv, read_vcard
    from brain.discovery.warm.paths import Network, annotate

    people = read_csv(LINKEDIN)
    assert [p["source"] for p in people] == ["linkedin", "linkedin"] and people[0]["domain"] == "voxa.ai"
    vc = read_vcard("BEGIN:VCARD\nFN:Sara Khan\nEMAIL;TYPE=work:sara@cloud.co\nORG:CloudCo;Eng\nEND:VCARD\n")
    assert vc[0]["name"] == "Sara Khan" and vc[0]["company"] == "CloudCo" and vc[0]["domain"] == "cloud.co"
    g = read_csv('"First Name","Last Name","E-mail 1 - Value","Organization Name"\nDev,Iyer,dev@acme.in,Acme\n')
    assert g[0]["source"] == "google_contacts" and g[0]["email"] == "dev@acme.in"
    store = Store(f"sqlite:///{tmp_path / 'w.db'}")
    store.network_add(people + vc)
    net = Network(store.network())
    known = Contact(kind="person", name="Priya Nair", org="Voxa AI", org_domain="voxa.ai")
    org = Contact(kind="org", name="Voxa AI", org="Voxa AI", org_domain="voxa.ai")
    media = Contact(kind="org", name="Paper Media", org="Paper Media")
    stranger = Contact(kind="org", name="Nobody Ltd", org_domain="nobody.io")
    stats = annotate([known, org, media, stranger], net)
    assert stats == {"direct": 1, "introduction": 2}
    assert known.signals["warm_path"] == 1.0 and known.evidence[0].text.startswith("You are already connected to Priya Nair (LinkedIn")
    assert org.signals["warm_path"] == 0.6 and "Priya Nair, Head of Engineering" in org.evidence[0].text
    assert "Arjun Mehta" in media.evidence[0].text and "warm_path" not in stranger.signals
    assert store.network_clear() == 3 and store.network_size() == 0


def test_warm_path_lifts_the_prescore():
    from brain.rank import prescore

    a, b = Contact(kind="org", name="A", org_domain="a.io", bio="developer tools"), Contact(kind="org", name="B", org_domain="b.io", bio="developer tools")
    b.signals["warm_path"] = 1.0
    ranked = prescore([a, b], _spec(target_kind="org", pack="sponsorship", topics=["developer tools"]), get_playbook("sponsorship"))
    assert ranked[0].name == "B"


# ── phase 8: playbooks and Hacker News ───────────────────────────────────────
def test_every_playbook_is_runnable():
    from brain.discovery.router import choose
    from brain.discovery.signals import known
    from brain.fit import profile_for

    assert len(PLAYBOOKS) >= 28
    for pb in PLAYBOOKS.values():
        assert all(s in TOOLS for s in pb.sources), pb.name
        assert all(known(c["signal"]) for c in pb.criteria + pb.timing), pb.name
        spec = _spec(pack=pb.name, target_kind=pb.target_kind)
        seed, expand = choose(spec, pb, profile_for(spec, pb), has_seeds=False)
        assert seed, pb.name


def test_hn_who_is_hiring_posts_become_companies():
    from brain.discovery.hooks.hn import parse_post

    post = ('Voxa AI | ML Engineer | Bengaluru, India (ONSITE) | Full-time<p>We build Indic speech models. '
            'Email founders at hiring [at] voxa [dot] ai or <a href="https://voxa.ai">voxa.ai</a>')
    p = parse_post(post)
    assert p["company"] == "Voxa AI" and p["site"] == "https://voxa.ai" and p["emails"] == ["hiring@voxa.ai"]
    idx = {"hits": [{"objectID": "900", "title": "Ask HN: Who is hiring? (November 2026)"}]}
    comments = {"hits": [{"objectID": "901", "parent_id": 900, "comment_text": post},
                         {"objectID": "902", "parent_id": 900, "comment_text": "FarCo | Eng | Berlin, Germany | ONSITE"},
                         {"objectID": "903", "parent_id": 901, "comment_text": "a reply"}]}
    f = FakeFetcher()
    f.get_json = lambda url, params=None, ttl=0, headers=None: idx if "search_by_date" in url else comments
    found, stats = TOOLS["hn_hiring"].fn(_run(_spec(target_kind="org", regions=["Bangalore"], countries=["IN"]), Deps(f), pb="job_search"))
    assert [c.name for c in found] == ["Voxa AI"] and found[0].org_domain == "voxa.ai"
    assert found[0].best_email.address == "hiring@voxa.ai" and found[0].best_email.method == "inbox"
    assert found[0].signals["hiring"] == 1.0 and stats["threads"] == 1


# ── phase 9: paid plugins ────────────────────────────────────────────────────
def test_paid_sources_only_with_a_key_and_counted(monkeypatch):
    from brain.discovery.adapters import collect
    from brain.discovery.registry import SPECS
    from brain.discovery.router import usable

    spec = _spec(target_kind="org")
    assert not usable(TOOLS["exa"], spec, get_playbook("general"))[0]
    monkeypatch.setenv("EXA_API_KEY", "k")
    assert usable(TOOLS["exa"], spec, get_playbook("general"))[0]
    f = FakeFetcher()
    f.paid_calls = {}
    posted = []
    f.post_json = lambda url, body, headers=None: posted.append((url, body, headers)) or {"results": [
        {"title": "Voxa AI", "url": "https://voxa.ai/", "text": "Indic speech models"}]}
    found, _ = collect(SPECS["exa"], _run(spec, Deps(f), pb="general"))
    assert [c.name for c in found] == ["Voxa AI"] and found[0].org_domain == "voxa.ai"
    assert posted[0][1]["query"] == "speech recognition companies India" and posted[0][2] == {"x-api-key": "k"}
    assert f.paid_calls == {"exa": 2}  # one per campaign term


def test_hunter_and_builtwith_respect_keys_and_caps(monkeypatch):
    from brain.discovery.reach import providers
    from brain.models import FitCriterion, ProspectProfile

    org = Contact(kind="org", name="Voxa", org_domain="voxa.ai")
    f = FakeFetcher()
    f.paid_calls = {}
    assert providers.hunter_person(f, org, ["CTO"]) is None  # no key, no call
    monkeypatch.setenv("HUNTER_API_KEY", "k")
    f.get_json = lambda url, params=None, ttl=0, headers=None: {"data": {"emails": [
        {"value": "kiran@voxa.ai", "type": "personal", "first_name": "Kiran", "last_name": "Rao", "position": "Engineer", "confidence": 90},
        {"value": "meera@voxa.ai", "type": "personal", "first_name": "Meera", "last_name": "Iyer", "position": "CTO", "confidence": 70},
        {"value": "info@voxa.ai", "type": "generic"}]}}
    p = providers.hunter_person(f, org, ["CTO"], cap=1)
    assert p.name == "Meera Iyer" and p.best_email.method == "hunter" and p.best_email.confidence == 0.7
    assert providers.hunter_person(f, org, ["CTO"], cap=1) is None and f.paid_calls == {"hunter_domain": 1}
    monkeypatch.setenv("BUILTWITH_API_KEY", "k")
    f.get_json = lambda url, params=None, ttl=0, headers=None: {"groups": [{"name": "analytics", "categories": [{"name": "Shopify"}]}]}
    prof = ProspectProfile(criteria=[FitCriterion(name="runs on Shopify", signal="uses_tech", terms=["shopify"])])
    assert providers.tag_tech([org], prof, f) == 1 and org.signals["uses_tech"] == 1.0
