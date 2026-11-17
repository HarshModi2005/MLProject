"""Discovery v3: source manifests, generic adapters, hooks, router, ledger, deterministic decision-makers,
per-source funnel and label sheets. No network: every external answer is scripted."""

import json

import pytest

from brain.context import CampaignContext
from brain.discovery import registry
from brain.discovery.contracts import SourceSpec
from brain.discovery.ledger import Ledger
from brain.discovery.normalize import campaign_terms, get_path, render
from brain.fetch import Page
from brain.models import CampaignSpec, Contact, EmailCandidate, SenderProfile, TargetPersona
from brain.playbooks import get_playbook
from brain.searchplan import SearchPlanner
from brain.sources.websearch import Hit, NewsHit
from brain.tools import TOOLS, ToolRun

FEED = """<?xml version="1.0"?><rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>
<title>AI in India</title><itunes:author>Meera Iyer</itunes:author><description>Weekly AI conversations</description>
<itunes:owner><itunes:name>Meera Iyer</itunes:name><itunes:email>Meera@AIindia.fm</itunes:email></itunes:owner>
<item><title>Speech models for Indic languages</title><link>https://aiindia.fm/1</link>
<pubDate>Mon, 02 Nov 2026 10:00:00 GMT</pubDate></item></channel></rss>"""


def _spec(**kw):
    base = dict(id="v3", pack="pr_media", objective="Coverage for a speech model launch", ask="Interview?",
                persona=TargetPersona(description="AI journalists", roles=["Journalist"]),
                topics=["speech recognition", "Indian language AI"], expanded_terms=["open source AI"],
                countries=["IN"], target_count=5, sender=SenderProfile(name="T"))
    base.update(kw)
    return CampaignSpec(**base)


class FakeCache:
    def __init__(self):
        self.d = {}

    def get(self, ns, key):
        return self.d.get((ns, key))


class FakeSettings:
    replay = ""
    use_memory = False
    fetch_workers = 2
    min_email_confidence = 0.6


class FakeFetcher:
    def __init__(self, pages=None, api=None):
        self.pages, self.api = pages or {}, api or {}
        self.json_calls, self.fetched = [], 0

    def get(self, url):
        html = self.pages.get(url)
        return Page(url, 200, html) if html else Page(url, 404, error="missing")

    def get_many(self, urls):
        return [self.get(u) for u in dict.fromkeys(urls)]

    def get_json(self, url, params=None, ttl=0, headers=None):
        self.json_calls.append((url, dict(params or {}), dict(headers or {})))
        key = (url, (params or {}).get("term") or (params or {}).get("q") or "")
        return self.api.get(key, self.api.get(url))


class FakeSearch:
    def __init__(self, web=None, news=None, budget=50):
        self.web, self.news_hits = web or {}, news or {}
        self.calls, self.budget, self.queries = 0, budget, []
        self.available = True

    @property
    def remaining(self):
        return self.budget - self.calls

    def search(self, q, n=8, raw=False):
        self.calls += 1
        self.queries.append(q)
        return self.web.get(q, [])

    def news(self, q, n=10, gl=""):
        self.calls += 1
        self.queries.append(q)
        return self.news_hits.get(q, [])


class Deps:
    def __init__(self, fetcher=None, search=None):
        self.fetcher, self.search = fetcher or FakeFetcher(), search or FakeSearch()
        self.cache, self.settings, self.llm, self.store = FakeCache(), FakeSettings(), None, None


def _run(spec, deps, pb="pr_media", winners=None):
    p = get_playbook(pb)
    ctx = CampaignContext()
    return ToolRun(spec, p, ctx, deps, SearchPlanner(spec, p, ctx, None), budget=None, winners=winners or [])


def _manifest(**kw):
    base = {"name": "t", "adapter": "json_api", "kinds": ["org"], "policy": {"robots": "api"}}
    base.update(kw)
    return SourceSpec(**base)


# ── manifests and registry ───────────────────────────────────────────────────
def test_builtin_manifests_load_and_keep_the_historical_tool_order():
    assert not [e for e in registry.errors() if "discovery/sources" in e[0]]
    names = list(TOOLS)
    legacy = ["openalex", "orgs", "places", "web", "github", "news", "events", "jobs", "portfolio", "podcasts",
              "lookalike", "coauthors", "memory"]
    assert [n for n in names if n in legacy] == legacy  # new sources slot in without reordering the old ones
    assert {"arxiv", "nsf_awards", "npm", "huggingface", "gdelt", "google_play"} <= set(names)
    assert TOOLS["podcasts"].spec is not None and not TOOLS["podcasts"].needs_search
    assert TOOLS["news"].needs_search and TOOLS["jobs"].needs_search and TOOLS["jobs"].kinds == ("org",)


def test_manifest_validation_rejects_unknown_signals_and_missing_policy(tmp_path):
    with pytest.raises(Exception):
        _manifest(proves=["telepathy"])
    with pytest.raises(Exception):
        SourceSpec(name="x", adapter="json_api")  # no policy
    with pytest.raises(Exception):
        _manifest(adapter="python")  # no entry
    ok = _manifest(proves=["mind_reading"], declare_signals=[{"name": "mind_reading", "description": "x"}])
    assert ok.proves == ["mind_reading"]
    (tmp_path / "bad.json").write_text(json.dumps({"name": "bad", "adapter": "nope", "policy": {}}))
    (tmp_path / "good.json").write_text(json.dumps({"name": "good_src", "adapter": "rss", "policy": {"robots": "respect"},
                                                    "request": {"url": "https://x.org/feed"}}))
    loaded = registry.load_dir(tmp_path)
    assert list(loaded) == ["good_src"] and any("bad.json" in p for p, _ in registry.errors())


def test_user_manifest_installs_as_a_tool(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"name": "user_src", "adapter": "rss", "kinds": ["person"],
                                                 "proves": ["recent_coverage"], "policy": {"robots": "respect"},
                                                 "request": {"url": "https://x.org/feed"}}))
    tools = dict(TOOLS)
    registry.install(tools, user_dir=tmp_path)
    assert "user_src" in tools and list(tools)[-1] == "user_src" and tools["user_src"].kinds == ("person",)


# ── templates ────────────────────────────────────────────────────────────────
def test_templates_paths_and_terms():
    data = {"a": {"b": [{"c": "X"}]}, "n": 25}
    assert get_path(data, "a.b.0.c") == "X" and get_path(data, "a.z") is None
    assert render("{a.b.0.c|lower}-{missing}", data) == "x-" and render(25, data) == 25
    assert render("q={t|quote}", {"t": "a b"}) == "q=a%20b"
    spec = _spec()
    assert campaign_terms(spec, {"terms": ["topics", "expanded_terms:1"], "max": 2}) == ["speech recognition",
                                                                                        "Indian language AI"]


# ── adapters ─────────────────────────────────────────────────────────────────
def test_json_api_generic_map_dedupe_and_auth(monkeypatch):
    monkeypatch.setenv("T_KEY", "secret")
    api = {("https://api.x/search", "speech recognition"): {"results": [
        {"id": 1, "company": {"name": "Voxa", "url": "https://voxa.ai/"}, "about": "Speech APIs", "mail": "hi@voxa.ai"},
        {"id": 2, "company": {"name": "Nope"}}]},
        ("https://api.x/search", "Indian language AI"): {"results": [
            {"id": 1, "company": {"name": "Voxa", "url": "https://voxa.ai/"}}]}}
    spec = _manifest(request={"url": "https://api.x/search", "params": {"q": "{term}", "limit": 5}},
                     query={"terms": ["topics"]}, items="results", dedupe="id", require=["company.url"],
                     map={"name": "company.name", "website": "company.url", "bio": "about", "email": "mail", "key": "id"},
                     signals={"hiring": 1.0}, evidence="Listed: {company.name}", proves=["hiring"],
                     auth={"env": "T_KEY", "header": "X-Key"})
    from brain.discovery.adapters import collect

    deps = Deps(FakeFetcher(api=api))
    found, stats = collect(spec, _run(_spec(), deps))
    assert [c.name for c in found] == ["Voxa"] and stats["requests"] == 2 and stats["items"] == 2
    c = found[0]
    assert c.kind == "org" and c.org_domain == "voxa.ai" and c.signals == {"hiring": 1.0} and "t:1" in c.keys
    assert c.best_email.address == "hi@voxa.ai" and c.evidence[0].text == "Listed: Voxa"
    assert deps.fetcher.json_calls[0][1] == {"q": "speech recognition", "limit": 5}  # ints stay ints
    assert deps.fetcher.json_calls[0][2] == {"X-Key": "secret"}


def test_podcasts_manifest_follows_feeds_to_the_hosts_email():
    itunes = {"results": [
        {"feedUrl": "https://feeds.x/ai", "collectionName": "AI in India", "collectionViewUrl": "https://pod.x/ai",
         "trackCount": 100, "genres": ["Technology", "Podcasts"]},
        {"feedUrl": "https://feeds.x/ai", "collectionName": "dupe"}, {"collectionName": "no feed"}]}
    deps = Deps(FakeFetcher(pages={"https://feeds.x/ai": FEED}, api={"https://itunes.apple.com/search": itunes}))
    found, stats = TOOLS["podcasts"].fn(_run(_spec(), deps))
    assert len(found) == 1
    c = found[0]
    assert c.kind == "person" and c.name == "Meera Iyer" and c.org == "AI in India" and c.sources == ["podcasts"]
    assert c.best_email.address == "meera@aiindia.fm" and c.best_email.method == "feed"
    assert c.signals["audience"] == 0.5 and c.topics == ["Technology"] and "channel_verified" in c.flags
    url, params, _ = deps.fetcher.json_calls[0]
    assert params == {"media": "podcast", "term": "speech recognition", "limit": 25, "country": "in"}
    assert len(deps.fetcher.json_calls) == 3  # two topics + one expanded term


def test_rss_adapter_reads_items_with_a_field_map():
    spec = SourceSpec(name="feedsrc", adapter="rss", kinds=["person"], proves=["recent_coverage"],
                      policy={"robots": "respect"}, request={"url": "https://news.x/rss?q={term|quote}"},
                      query={"terms": ["topics"], "max": 1}, map={"name": "channel.author", "url": "item.link"},
                      evidence="Wrote: {item.title}", signals={"recent_coverage": 1.0})
    from brain.discovery.adapters import collect

    deps = Deps(FakeFetcher(pages={"https://news.x/rss?q=speech%20recognition": FEED}))
    found, stats = collect(spec, _run(_spec(), deps))
    assert stats["feeds"] == 1 and found[0].name == "Meera Iyer" and found[0].evidence[0].text.startswith("Wrote: Speech")


def test_list_page_and_sitemap_read_cards_with_selectors():
    html = """<div class="m"><h3>Sunbeam Solar</h3><a class="w" href="https://sunbeam.in/">site</a>
    <a href="mailto:sales@sunbeam.in">mail</a><p>Rooftop solar dealer in Ahmedabad</p></div>
    <div class="m"><h3>Ravi Energy</h3><p>EPC company</p></div>"""
    sel = {"item": ".m", "fields": {"name": "h3", "website": "a.w@href", "email": "a[href^='mailto:']@href", "bio": "p"}}
    common = dict(kinds=["org"], policy={"robots": "respect"}, select=sel, signals={"member_of": 1.0}, proves=["member_of"],
                  map={"name": "name", "website": "website", "email": "email", "bio": "bio"})
    from brain.discovery.adapters import collect

    lp = SourceSpec(name="members", adapter="list_page", request={"urls": ["https://assoc.in/members?p={page}"],
                                                                 "pages": [1]}, **common)
    deps = Deps(FakeFetcher(pages={"https://assoc.in/members?p=1": html}))
    found, _ = collect(lp, _run(_spec(), deps))
    assert [c.name for c in found] == ["Sunbeam Solar", "Ravi Energy"]
    assert found[0].org_domain == "sunbeam.in" and found[0].best_email.address == "sales@sunbeam.in"

    sm = SourceSpec(name="teams", adapter="sitemap", stage="expand",
                    request={"path": "/dealers/"}, **common)
    xml = "<urlset><url><loc>https://win.in/dealers/a</loc></url><url><loc>https://win.in/blog/x</loc></url></urlset>"
    deps = Deps(FakeFetcher(pages={"https://win.in/sitemap.xml": xml, "https://win.in/dealers/a": html}))
    found, stats = collect(sm, _run(_spec(), deps, winners=[Contact(kind="org", name="Win", org_domain="win.in")]))
    assert stats["pages"] == 1 and len(found) == 2


def test_search_adapter_templates_and_news_hooks():
    from brain.discovery.adapters import collect

    spec = SourceSpec(name="play", adapter="search", kinds=["org"], proves=["app_publisher"], cost="search",
                      policy={"robots": "respect"}, query={"terms": ["topics"], "max": 1},
                      search={"templates": ["site:play.google.com {term}"], "match": "play.google"},
                      map={"name": "title", "url": "url"}, signals={"app_publisher": 1.0})
    web = {"site:play.google.com speech recognition": [Hit("https://play.google.com/store/apps/details?id=a", "Voxa"),
                                                       Hit("https://other.com/x", "Other")]}
    deps = Deps(search=FakeSearch(web=web))
    found, stats = collect(spec, _run(_spec(), deps))
    assert [c.name for c in found] == ["Voxa"] and deps.search.queries == ["site:play.google.com speech recognition"]

    article = """<html><head><meta name="author" content="Priya Nair"></head><body>Story</body></html>"""
    news = {"speech recognition India": [NewsHit("https://paper.in/a1", "Speech AI for Indic", "s", "The Paper", "1d"),
                                         NewsHit("https://paper.in/a2", "More speech AI", "s", "The Paper", "2d")]}
    deps = Deps(FakeFetcher(pages={"https://paper.in/a1": article, "https://paper.in/a2": article}), FakeSearch(news=news))
    found, _ = TOOLS["news"].fn(_run(_spec(), deps))
    assert [c.name for c in found] == ["Priya Nair"] and found[0].title == "Journalist"
    assert [e.text for e in found[0].evidence] == ["Speech AI for Indic", "More speech AI"]  # a repeat adds evidence


# ── router ───────────────────────────────────────────────────────────────────
def test_router_skips_sources_without_keys_or_outside_their_region(monkeypatch):
    from brain.discovery.router import choose, skipped, usable
    from brain.fit import profile_for

    keyed = registry.tool_for(_manifest(name="keyed", auth={"env": "NO_SUCH_KEY_V3"}))
    us_only = registry.tool_for(_manifest(name="us_only", regions=["US"]))
    off = registry.tool_for(_manifest(name="off", enabled_by_default=False))
    spec = _spec()
    pb = get_playbook("pr_media")
    assert usable(keyed, spec, pb) == (False, "needs NO_SUCH_KEY_V3")
    assert not usable(us_only, spec, pb)[0] and usable(us_only, _spec(countries=[]), pb)[0]
    assert not usable(off, spec, pb)[0]
    monkeypatch.setenv("NO_SUCH_KEY_V3", "x")
    assert usable(keyed, spec, pb)[0]
    seed, expand = choose(spec, pb, profile_for(spec, pb), has_seeds=False)
    assert seed == ["memory", "news", "gdelt", "podcasts"] and skipped(spec, pb) == []


# ── ledger ───────────────────────────────────────────────────────────────────
def test_ledger_counts_spend_and_yield_per_source():
    deps = Deps()
    led = Ledger()
    with led.measure("news", deps) as e:
        deps.search.calls += 3
        deps.fetcher.fetched += 5
        e.update(found=4, qualified=2)
    with led.measure("news", deps) as e:
        e["found"] = 1
    r = led.report()["news"]
    assert r["runs"] == 2 and r["search_calls"] == 3 and r["fetches"] == 5 and r["found"] == 5 and r["qualified"] == 2


# ── deterministic decision-maker searches ─────────────────────────────────────
def test_decision_maker_searches_go_to_organisations_in_rank_order(monkeypatch):
    from brain.orgs import OrgEnricher

    class S:
        remaining = 50

    oe = OrgEnricher(FakeFetcher(), S(), FakeSettings(), find_people=True, max_people_searches=2)
    orgs = [Contact(kind="org", name=f"Org{i}", org_domain=f"org{i}.com") for i in range(5)]

    def fake_enrich(org, spec, roles, defer_dm=False):
        oe._pending[id(org)] = ([], {}, None)
        return org

    asked = []

    def fake_dm(org, roles, people):
        asked.append(org.name)
        return None

    monkeypatch.setattr(oe, "enrich", fake_enrich)
    monkeypatch.setattr(oe, "_find_decision_maker", fake_dm)
    out = oe.enrich_all(orgs, _spec(target_kind="org"), get_playbook("sponsorship"))
    assert asked == [f"Org{i}" for i in range(5)] and [o.name for o in out] == [f"Org{i}" for i in range(5)]


# ── funnel and labels ────────────────────────────────────────────────────────
def test_source_funnel_counts_each_source():
    from brain.pipeline import count_sources, note_sources

    cs = [Contact(name="A B", sources=["news", "podcasts"]), Contact(name="C D", sources=["news"]), Contact(name="E F")]
    assert count_sources(cs) == {"news": 2, "podcasts": 1, "unknown": 1}
    f = {}
    note_sources(f, "unique", cs[:1])
    note_sources(f, "unique", cs[1:2], add=True)
    assert f["source_funnel"]["unique"] == {"news": 2, "podcasts": 1}


def test_label_sheets_export_and_feed_the_bench(tmp_path, monkeypatch):
    import brain.bench as bench
    from brain.store import Store

    monkeypatch.setattr(bench, "RESULT_DIR", tmp_path)
    (tmp_path / "run_r1").mkdir()
    store = Store(f"sqlite:///{tmp_path / 'run_r1' / 'sc.db'}")
    good = Contact(id="g", name="Priya Nair", org="The Paper", sources=["news"],
                   emails=[EmailCandidate(address="priya@paper.in", confidence=0.9, method="text")])
    bad = Contact(id="b", name="Some Show", kind="org", sources=["podcasts"])
    store.checkpoint("x", "enrich", [good, bad])
    labels = tmp_path / "labels"
    path = bench.export_label_sheet("r1", "sc", folder=labels)
    sheet = json.loads(path.read_text())
    assert [c["name"] for c in sheet["candidates"]] == ["Priya Nair", "Some Show"]
    sheet["candidates"][0]["label"] = "good"
    sheet["candidates"][1]["label"] = "bad"
    path.write_text(json.dumps(sheet))
    assert bench.load_labels("sc", labels) == {"good": ["priya@paper.in"], "bad": ["Some Show"]}
    bench.export_label_sheet("r1", "sc", folder=labels)  # re-export keeps the labels
    assert bench.load_labels("sc", labels)["good"] == ["priya@paper.in"]
