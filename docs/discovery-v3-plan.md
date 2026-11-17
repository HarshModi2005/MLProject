# Discovery v3 — plan

**Goal:** discovery that can reach any public source relevant to any outreach use case, without losing
precision or blowing the budget, built so that adding a source or a use case is configuration, not surgery.

Status: Phases 0–1 implemented and committed (c046ace); Phase 2 committed (d3a5065); Phase 3 wave 1 in progress (2026-11-17) · Branch base: `discovery-v2` · Owner: discovery module (`brain/`)

### Progress

| Phase | State | Notes |
|---|---|---|
| 0 Measurement | done, except the labels themselves | per-source funnel (`source_funnel`, `ledger` in every bench row and table), backfill counted in `unique_people`, label sheets (`brain bench-labels --run <label>` → `evals/labels/<scenario>.json`, read by `brain bench`), 4 new scenarios (`grants_csr_india`, `speakers_ai_summit`, `b2b_app_devs`, `distributors_solar`; not yet run live). **Human labels still to fill in.** |
| 1 Source framework | done | `brain/discovery/` with all five generic adapters plus `python`; `podcasts` (json_api → rss) and `news` (search + hooks) are generic manifests; `jobs` is a manifest using the `python` adapter (its three ATS APIs move to generic manifests in Phase 3). Router replaces `choose_tools`. Replay parity: discover and rank stages identical on all 6 scenarios. |
| 2 Precision layer | built; end-to-end effect not yet measured live | `precision/orgtype.py` (org-type classes + gate, `Playbook.allowed_org_classes`), `precision/prejudge.py` (one batched fast call on the top ~40 before enrichment; rejected candidates are never enriched), `precision/fitmodel.py` (examples stored after every judge, logistic model per playbook once ≥50 examples, used by the explorer's "qualified"), `precision/priors.py` (per-source history orders paid seed sources and feeds the bandit), byline filter for organisation names. **Screen check** (`brain bench-screen --run p2_before`, same candidates as the judge): 0/19 judge-good and 0/8 middling rejected, 9/19 judge-bad caught. End-to-end replay fell 19 → 13 good@K, but replay misses tripled (new candidates were never recorded), so only a live run can measure the end-to-end effect. |
| 3 Source wave 1 | partly done | **Live-verified:** `nsf_awards` (US researchers + their listed email; 25/25 with email on a test query), `npm` (maintainers + registry email; bot/typings accounts filtered), `huggingface` (model authors; no email), `arxiv` (authors kept only when their paper's PDF prints their address; 11 of 32 authors on a test query; ~2 min because arXiv's Crawl-delay is 15 s), `google_play` (developer, website, email from app pages; off unless a criterion needs `app_publisher`), jobs now also reads Workable, Recruitee and SmartRecruiters (two grouped `site:` searches per topic). **Built, not verified live:** `gdelt` (free news → bylines; the API rate-limited this machine during testing). Infrastructure: Crawl-delay honoured, `Fetcher.get_pdf_text` / `get_api_text` (cached, replayable), `reach/pdfemails.py`, XML APIs (`request.format: text` + `parse` hook), `post` hooks, `{x|prefix=..}` / `{x|default=..}` filters, date variables, inbox rules for published emails. **Not yet:** Sessionize, exhibitor lists, YC directory, Indian ATS (Keka / Darwinbox / Zoho: no verified public API), csr.gov.in, Grants.gov / NIH / CORDIS, Crossref / Semantic Scholar / DBLP, HN, Wikidata, company registries, tenders, associations, Chrome Web Store, GitHub commit emails, Podcast Index (needs a key). |
| Fix found on the way | done | Decision-maker searches raced between threads (same code gave 8, 9 or 10 decision-makers across replays). They now run in rank order after the parallel site reads: deterministic, and the best-ranked organisations get the capped searches. |

Baselines: `evals/results/pre_v3_replay.json` (before), `evals/results/v3_replay_a.json` (after Phases 0–1), both
`brain bench --replay --strict`.

### Adding a source (now)

1. Write `sources/<name>.json` (user) or `brain/discovery/sources/<name>.json` (built-in): see the contract in
   `brain/discovery/contracts.py` and the adapter docstrings in `brain/discovery/adapters/`.
2. If a field map cannot express it, add `brain/discovery/hooks/<name>.py` with `extract` / `to_contact` /
   `key` / `merge` and point `"hooks"` at it.
3. New kind of proof? Add it to `brain/discovery/signals.py` (or `declare_signals` in the manifest).
4. Add a fixture test in `tests/brain/test_discovery_v3.py`; list the source in a playbook's `sources`, or let
   the router add it when a criterion needs the signal it proves.

---

## 1. Principles

1. **Sources are plugins, described as data.** One manifest per source. Most sources need zero Python
   (generic adapters); complex ones ship a small adapter class behind the same contract.
2. **Recall is cheap, precision is learned.** Free/structured sources run wide (Tier 1); paid search,
   page reads and the LLM run narrow on the best candidates (Tier 2). Widening only ships behind the
   precision layer (Phase 2), because v2 showed that more sources without it lowers quality
   (sponsors: events-only 5/6 vs. mixed 1–3/6).
3. **Reachability is a first-class output.** Every source declares whether it yields contact details;
   the router prefers sources that do. A great prospect with no email is a failure.
4. **Everything compounds.** Every entity, edge, email pattern, query and judge verdict is stored and
   reused by later campaigns (persistent index, learned priors).
5. **Provenance and policy on every record.** Each fact carries its source URL and the source's usage
   policy; nothing is collected that the source's terms or robots.txt forbid.
6. **Backwards compatible.** New sources register into the existing `TOOLS` registry, so
   `explore.py`, `pipeline.py` and playbooks keep working at every step.

## 2. Target architecture

```
                         CampaignSpec + ProspectProfile (intake)
                                         │
                                  ┌──────▼──────┐
                                  │   Router     │  picks sources by kind, signals needed, region,
                                  │ (router.py)  │  reachability, cost, learned priors per playbook
                                  └──────┬──────┘
          ┌──────────────────────────────┼──────────────────────────────┐
   Tier 1 (wide, free)            Tier 2 (narrow, paid)          Memory (free, instant)
   JSON/RSS/list/sitemap          web/news/places search          persistent index + graph
   adapters over manifests        light reads, extraction         (index.py, graph.py)
          └──────────────────────────────┼──────────────────────────────┘
                                  ┌──────▼──────┐
                                  │ Normalize    │  Record -> Contact, Evidence, signals, provenance
                                  └──────┬──────┘
                                  ┌──────▼──────┐
                                  │ Resolve +    │  entity resolution, graph edges
                                  │ Graph write  │
                                  └──────┬──────┘
                                  ┌──────▼──────┐
                                  │ Expand       │  graph walks: sponsor→events→sponsors, investor→
                                  │ (walkers)    │  portfolio→founders, journalist→outlet→beat ...
                                  └──────┬──────┘
                                  ┌──────▼──────┐
                                  │ Precision    │  learned fit model, org-type gate, pre-judge
                                  └──────┬──────┘
                                  ┌──────▼──────┐
                                  │ Reach        │  published emails, domain patterns, author pages,
                                  │ (reach/)     │  paper PDFs, MX, optional verifier
                                  └──────┬──────┘
                                     judge → shortlist → draft (unchanged)
```

## 3. Module layout

New package `brain/discovery/`. Existing modules stay; v2 code moves in gradually behind shims.

```
brain/discovery/
  __init__.py
  contracts.py      SourceSpec, Query, Record, SourceResult, Adapter protocol
  registry.py       loads manifests (built-in + ./sources/*.json) and Python sources; registers into TOOLS
  router.py         choose sources for a spec (replaces choose_tools; same output shape)
  signals.py        signal registry (replaces fit.SIGNAL_KEYS): name, description, decay, proof strength
  normalize.py      Record -> Contact using the manifest's field map; evidence + signals + provenance
  policy.py         per-source usage policy: robots, ToS flags, allowed fields, retention, rate limits
  ledger.py         cost ledger: calls / fetches / LLM per source, per campaign, per tier
  adapters/
    json_api.py     URL template, params, auth, pagination, JSONPath field mapping
    rss.py          RSS/Atom (podcasts, Substack, Google News, journals, blogs)
    list_page.py    CSS-selector extraction from directory / member / exhibitor / sponsor pages
    sitemap.py      sitemap.xml + path patterns (/team/, /speakers/, /portfolio/) on a domain
    search.py       search verticals (web, news, places) + site: templates; engine-agnostic
  sources/          built-in manifests (*.json) and the few Python adapters that need code
  graph.py          entity graph: nodes (org, person, event, article, repo, paper, fund, grant), edges
  walkers.py        graph-walk expansion tools (generalises lookalike + coauthors)
  index.py          persistent knowledge base: upsert, query by profile, freshness, refresh jobs
  precision/
    fitmodel.py     learned criterion weights + threshold per playbook (from judge verdicts / labels)
    orgtype.py      org-type classifier at light read (academic, media, publisher, directory, ...)
    prejudge.py     batched fast-model fit check before enrichment
  reach/
    patterns.py     cross-campaign domain email-convention table + size-based priors
    authorpages.py  journalist / author page reader
    pdfemails.py    corresponding-author emails from paper PDFs (PyMuPDF, already a dependency)
    published.py    emails from app-store listings, package metadata, feeds, commits
    verify.py       MX check (exists) + optional paid verifier plugin
  warm/             user-owned data connectors (mail contacts, LinkedIn export, CRM, CSV)
```

User-supplied manifests live in `./sources/*.json`, like user playbooks in `./playbooks`.

## 4. Core contracts

### 4.1 Source manifest (JSON, same format as user playbooks)

```json
{
  "name": "google_play_dev",
  "title": "App developers on Google Play",
  "adapter": "list_page",
  "tier": 1,
  "cost": "fetch",
  "kinds": ["org"],
  "stage": "seed",
  "regions": ["*"],
  "proves": ["app_publisher", "topic"],
  "yields_email": true,
  "query": {"via": "search", "template": "site:play.google.com/store/apps {topic}"},
  "extract": {
    "name": "[itemprop=author] span", "website": "a[href^='http']:contains('Website')",
    "email": "a[href^='mailto:']", "about": "[itemprop=description]"
  },
  "signals": {"app_publisher": 1.0},
  "evidence": "Publishes the app {title} on Google Play",
  "policy": {"robots": "respect", "fields": ["name", "website", "email", "about"], "rate_per_min": 20},
  "auth": null,
  "enabled_by_default": true
}
```

Required fields: `name, adapter, kinds, proves, yields_email, cost, tier, policy`.
Validation at load time (pydantic `SourceSpec`); a bad user manifest is skipped with a warning.

### 4.2 Adapter protocol

```python
class Adapter(Protocol):
    def plan(self, spec: SourceSpec, campaign: CampaignSpec, seeds: list[Contact]) -> list[Query]: ...
    def fetch(self, q: Query, run: ToolRun) -> list[Record]: ...   # network; uses Fetcher/WebSearch + cache
```

`Record` is a flat dict + provenance (`source`, `url`, `fetched_at`). `normalize.py` turns records into
`Contact`s with the manifest's field map, signals and evidence template. Every adapter goes through the
existing `Fetcher` (per-host delay, cache, replay) and `WebSearch` (budget), so the bench's replay mode
works for every new source automatically.

### 4.3 Registration

`registry.load_all()` turns each manifest into a `Tool` in the existing `TOOLS` dict (name, `proves`,
`kinds`, `stage`, `needs_search` derived from `query.via`). Playbooks keep listing source names.
Python-only sources keep using `@register`.

### 4.4 Signals

`signals.py` holds the signal registry: built-ins from `fit.SIGNAL_KEYS` plus new ones
(`app_publisher, uses_tech, recently_funded, open_tender, grant_awarded, exhibits_at, maintains_package,
warm_path, accelerator_alum, csr_spend`). A manifest may only use registered signals; adding one is one
entry. `fit.py`, the judge card (`_PROOF_KEYS`) and the bench's `proof_rate` read from the registry.

### 4.5 Entity graph

Tables (SQLAlchemy, same DB as `Store`): `graph_nodes(id, type, key, name, data_json, first_seen, last_seen)`,
`graph_edges(src, dst, type, weight, source, url, seen_at)`. Edge types: `sponsored, spoke_at, exhibited_at,
wrote_for, covers, invested_in, founded, works_at, coauthored, maintains, competes_with, hiring_for,
awarded, member_of`. Contacts keep their ids (`resolve.assign_id`), so a node id = contact id.

## 5. Phases

Every phase ends with tests plus a replay bench run against the previous phase. No phase may lower
labelled precision@K on any scenario. Live bench runs stay small (one or two scenarios) to save tokens.

### Phase 0 — Measurement foundation  *(prerequisite)*
- Hand-label ~20 candidates per scenario (`labels: {good, bad}`); `evalkit` already scores them.
- Bench reports `label_precision@K` as the headline metric, judge good@K second.
- Fix funnel accounting: `unique_people` includes backfill finds (`pipeline.py:162`).
- Add per-source metrics to the bench: candidates, qualified, reachable, passed-judge **per source**.
- Add 4 scenarios for new use cases (co-founder search, distributors, grants/CSR, app-developer B2B).
- **Exit:** every scenario labelled; per-source funnel visible in `brain bench` output.

### Phase 1 — Source framework (no behaviour change)
- `contracts.py`, `registry.py`, `normalize.py`, `policy.py`, `ledger.py`, `signals.py`.
- Generic adapters: `json_api`, `rss`, `list_page`, `sitemap`, `search`.
- Port 3 existing tools onto manifests to prove the design: `podcasts` (json_api + rss), `jobs`
  (search + json_api per ATS), `news` (search vertical + byline extractor hook).
- `router.py` reproduces `choose_tools` exactly, then adds region / `yields_email` / cost filters.
- **Exit:** replay bench identical (±0) to v2 on all scenarios; unit tests for each adapter with
  fixtures in `tests/fixtures/sources/`.

### Phase 2 — Precision layer  *(must land before widening)*
- `precision/orgtype.py`: heuristics (`.ac.*`/`.edu` → academic; news domains → media; publisher,
  directory, aggregator patterns) + batched fast-LLM fallback; gate on `persona.org_types`.
- `precision/prejudge.py`: after the light read, one batched fast call on the top ~40; only passes are
  enriched. Moves the judge in front of the expensive stage (`pipeline.py:271-317`).
- `precision/fitmodel.py`: store (features, judge score, label) per candidate in a new
  `brain_fit_examples` table; fit per-playbook criterion weights + qualified threshold (logistic
  regression, numpy only). `explore.qualified()` and `prescore` use the learned model when ≥50 examples.
- Source priors: per-playbook "passed-judge per search call" for each source, stored across campaigns;
  the router and the UCB bandit start from these instead of zero.
- **Exit:** label precision@K up on sponsors and jobs; enrichment searches per good result down.

### Phase 3 — Source wave 1: free and reachable
Favour sources that publish contact details. Each is a manifest unless noted (py = Python adapter).

| Source | Adapter | Proves | Email? | Use cases |
|---|---|---|---|---|
| Paper PDFs (arXiv, open-access PDFs from OpenAlex) | py (`reach/pdfemails`) | publications | yes | academic, speakers, PhD |
| Author / journalist pages | py (`reach/authorpages`) | recent_coverage | often | pr_media |
| Google Play developer listings | list_page | app_publisher | yes | b2b, partnerships |
| Chrome Web Store / WordPress plugins | list_page / json_api | app_publisher | yes | b2b, devtools |
| npm / PyPI / Hugging Face Hub | json_api | maintains_package, open_source | often | recruiting, devtools, co-founder |
| GitHub commit authors (org repos) | py | open_source | yes | jobs, recruiting |
| YC company directory | json_api | accelerator_alum, hiring, founders | no (names) | jobs, b2b, partnerships |
| Indian ATS boards: Keka, Darwinbox, Zoho Recruit; Workable, SmartRecruiters, Recruitee | search + json_api | hiring | no | jobs |
| Sessionize speaker profiles | list_page | spoke_before | sometimes | speakers |
| Trade-show exhibitor lists | list_page | exhibits_at | no | b2b, distributors, sponsors |
| Podcast Index API | json_api + rss | audience, recent_coverage | yes (feed) | pr, creators |
| YouTube Data API (channel descriptions) | json_api | audience | often | creators, brand deals |
| GDELT + Google News RSS | json_api / rss | recent_coverage, recently_funded | no | pr, investors, timing |
| csr.gov.in CSR spend | json_api / list_page | csr_spend, funding_focus | no | grants |
| Grants.gov, NIH RePORTER, NSF Awards, CORDIS | json_api | grant_awarded | no | grants, academic |
| Crossref, Semantic Scholar, DBLP | json_api | publications | no | academic |
| Hacker News (Algolia), DEV.to | json_api | topic, open_source | rarely | customer research, recruiting |
| Wikidata SPARQL | json_api | topic, location | no | org universe seeding |
| UK Companies House, SEC EDGAR, OpenCorporates (free tier) | json_api | registered, directors | no | b2b, investors |
| GeM / CPPP / SAM.gov / TED tenders | list_page / json_api | open_tender | no | government |
| Industry associations (NASSCOM, CII, FICCI member lists) | list_page | member_of | sometimes | b2b, partnerships |

- **Exit:** each source has a fixture test, a policy block, and appears in the per-source bench funnel;
  the router enables it only for kinds and regions it serves.

### Phase 4 — Entity graph and graph-walk expansion
- `graph.py` writes nodes and edges during normalize/resolve (edges come from the manifests: a sponsor
  record yields `org -sponsored-> event`).
- `walkers.py` registers expansion tools (stage `expand`):
  `sponsor_walk` (sponsor → events → co-sponsors), `portfolio_walk` (investor → portfolio → founders),
  `beat_walk` (journalist → outlet → colleagues on the beat), `coauthor_walk` (generalised coauthors),
  `competitor_walk` (lookalike via graph + Exa-style "similar" search when configured),
  `alumni_walk` (accelerator cohort → peer companies).
- The explorer's adaptive rounds pick walkers like any other tool (UCB + priors).
- **Exit:** at least half of qualified finds in sponsors/investors/pr come from walks at lower search cost.

### Phase 5 — Reachability engine
- `reach/patterns.py`: `brain_domain_patterns(domain, pattern, support, last_confirmed)` filled from
  every confirmed email in every campaign; size priors (startup → `first@`, large → `first.last@`) as a
  low-confidence tier; MX required.
- Playbook flag `accept_guessed: bool` (jobs, pr: yes; academic: no) and a distinct method
  `pattern_prior` so drafts and UI can say "guessed".
- Decision-maker path uses the pattern table, not only site-confirmed patterns (`orgs.py:377`).
- `reach/verify.py`: optional verifier plugin interface (off by default; a paid provider plugs in).
- **Exit:** reachable rate among pre-judge passes ≥ 60% on pr and jobs scenarios.

### Phase 6 — Persistent index and refresh
- `index.py` generalises `Store.known_prospects`: query by profile criteria, region, signals, freshness.
- Refresh jobs (CLI `brain refresh`, cron-friendly): re-read known sponsor pages, ATS boards, feeds and
  portfolios; decay stale signals (`signals.py` decay per signal: hiring 30 days, coverage 90 days).
- Every campaign queries the index first (free round 0), then sources.
- **Exit:** second campaign in the same niche reaches the same good@K with ≤50% of the search calls.

### Phase 7 — Warm paths (user-owned data)
- `warm/` connectors: CSV (exists), LinkedIn connections export, Gmail/Outlook contacts and past threads
  (OAuth, read-only, local processing), HubSpot / Google Sheets CRM import.
- New signal `warm_path` (mutual connection / prior thread) with high weight; the draft step can mention
  the connection.
- Explicit consent and per-connector opt-in at intake; data never leaves the user's store.
- **Exit:** warm-path candidates rank first when present; no warm data used without opt-in.

### Phase 8 — Source wave 2 and new playbooks
- Remaining sources (Product Hunt, Meetup/Luma/Devfolio/Unstop event pages, Substack, Clutch/G2,
  IndiaMART/TradeIndia, Justdial, NGO Darpan, Candid, import/export records, UDISE+ school directory).
- New playbooks (JSON, mostly config once Phases 1–4 exist):
  `cofounder, advisors, accelerators, distributors, government_tenders, integrations, seo_guest_posts,
  brand_deals, campus_recruiting, school_outreach, phd_positions, awards_cfps, volunteering, alumni`.
- Each playbook ships with one bench scenario.

### Phase 9 — Paid plugins (opt-in per user)
Crunchbase/Tracxn, BuiltWith (`uses_tech`), Muck Rack, Exa, email verifier, X API. Each is a manifest
with `auth` and `cost` set; the ledger shows the money spent per campaign; off unless a key is present.

## 6. Use case → sources

| Use case (playbook) | Seed sources | Expansion walks | Reach |
|---|---|---|---|
| academic / phd_positions | openalex, crossref, semantic scholar, grants | coauthor_walk | pdf emails, orcid, faculty pages |
| sponsorship | events, exhibitor lists, csr | sponsor_walk, competitor_walk | inbox + decision-maker patterns |
| job_search | ATS (all), YC, GitHub orgs, funding news | alumni_walk, competitor_walk | commit emails, pattern priors |
| fundraising | portfolio, funding news, Form D, angel networks | portfolio_walk | fund site, patterns |
| pr_media | news, author pages, podcast index, substack | beat_walk | author pages, outlet patterns |
| creators / brand_deals | youtube, podcasts, substack | sponsor_walk (sponsors of similar creators) | channel/feed emails |
| b2b_sales / distributors | places, associations, exhibitors, app stores, registries | competitor_walk | site inbox, listing emails |
| recruiting / cofounder | github, npm/pypi, HF, hackathon winners | coauthor/maintainer walk | commit/profile emails |
| grants | csr.gov.in, grants.gov, candid, foundation reports | grantee walk (who else they fund) | program-officer pages |
| government_tenders | GeM, CPPP, SAM.gov, TED | buyer walk | official directories |
| speakers | sessionize, events, openalex | spoke_at walk | profile pages |
| warm (any) | user contacts, LinkedIn export, CRM | mutual-connection walk | known addresses |

## 7. Compliance

- `policy.py` enforces per-source: robots.txt, rate limit, allowed fields, retention; manifests without a
  policy block do not load.
- No scraping of sites whose terms forbid it (LinkedIn: search-result titles only, as now; Google
  Scholar: excluded). Reddit/X only through their APIs, paid tiers as opt-in plugins.
- Only publicly published or user-provided addresses; provenance (`source_url`) kept for every email;
  suppression/opt-out list (exists) checked before shortlist. DPDP (India), GDPR and CAN-SPAM notes in
  the README for deployers.

## 8. Testing

- Unit: each adapter against saved fixtures (`tests/fixtures/sources/<name>/`), normalize mappings,
  router choices per playbook, graph walkers on a synthetic graph, fit model training on fixtures.
- Contract test: every manifest loads, validates, has a policy, and only uses registered signals.
- Bench: replay mode for every change; live runs only for one or two scenarios at a time.

## 9. Risks

| Risk | Mitigation |
|---|---|
| More sources → more noise | Phase 2 ships first; per-source passed-judge priors throttle weak sources |
| Fragile CSS selectors in list_page manifests | fixture tests + a "selector health" stat in the ledger; LLM extraction fallback within budget |
| Cost creep | ledger per source and tier; Tier 1 free by default; paid plugins opt-in |
| Legal exposure when deployed | policy layer, provenance, suppression, public-data-only rule |
| Graph/index size in SQLite | indexes on (type, key); move to Postgres via `DATABASE_URL` when needed |

## 10. Open decisions

1. Exa / Brave as a second search engine in Phase 3, or wait for Phase 9 (cost)?
2. Should `accept_guessed` emails be allowed by default for jobs and pr (with a "guessed" flag)?
3. Warm paths (Phase 7): which connector first, Gmail or LinkedIn export?
4. Refresh jobs: local cron only, or a hosted scheduler once deployed?
