# No-repeat bulk outreach: "find me more" without ever contacting the same person twice

Goal: a user can keep asking for more recipients ("20 found, now find 20 more", repeated without bound) and every batch
contains only people who have not been surfaced, drafted, rejected or emailed before, while quality stays as high as
the first batch. When the pool for the brief really is exhausted, the engine says so instead of padding.

> **Status (2026-11-20): implemented on branch `no-repeat-bulk`, phases P0-P3, 244 offline tests pass** (23 new in
> `tests/brain/test_no_repeat.py`). Not exercised live (no search/LLM spend); the Streamlit review UI was only
> syntax-checked, not clicked through. Where the build differs from the plan below:
>
> - **Not built:** source cursors/pagination (rung "pages/cursors" of 2.5; the contract change needs a per-source audit) and
>   the `fresh_yield@batch_k` / `cost_per_fresh_good` bench metrics. The fresh yield is reported per batch in the funnel
>   (`supply.fresh_yield`) instead, so it can be tuned from real runs.
> - **Widening is opt-in:** by default `more` climbs only new queries then lookalikes. Neighbouring topics (`--widen topics`)
>   and a wider place (`--widen geo`) change what the user asked for, so they need to be asked for.
> - **One ledger row per identity key**, so across *different* series the latest one to touch a person holds the row (hard
>   states - sent, rejected, bounced, imported - are never overwritten). Within a series the behaviour is exactly as designed.
> - **Recycling lives in two places:** `brain more` offers it after the ladder (its own campaign, one clean review batch);
>   a direct `run_campaign` on a series also offers it when the batch is short. A batch where *nobody* new is found goes
>   through `more`, not the pipeline.
> - **The send gate adds a 30 s minimum gap** between sends (`BRAIN_MIN_SEND_DELAY`, 0 disables) because the old rate limiter
>   was never wired in; the review UI now stamps `sent_at`, which the cap and the follow-up engine count from.
> - **Known gap (pre-existing):** `scheduler/followup_engine.py` sends follow-ups directly and does not go through
>   `bridge.send_gate`, so follow-ups ignore the daily cap, the minimum gap and a paused series. Wiring it needs a
>   follow-up mode of the gate that tolerates "already emailed"; not done here.
> - **Code review:** a `/code-review high` pass found and this branch fixes: `series-resume` re-pausing at once (the bounce
>   rate now counts from the resume), the send gate refusing people offered again after a cooldown, stale review rows
>   resurrecting recycled people, sends not reaching the ledger after another series took the row, `--yes` silently
>   enabling recycling, a skipped recycle candidate being drafted anyway, and a do-not-contact import downgrading a send.
> - Code map: `brain/touch.py` (identity + policy), `brain/store.py` (ledger, series, claim, recycle, health),
>   `brain/series.py` (`more`, ladder, `until`), `brain/pipeline.py` (scope, early blocking, claim, recycle, supply funnel),
>   `brain/bridge.py` (review sync, send gate, recycled-draft move), `ui/review_ui.py`, `brain/cli.py`.

## 1. What happens today (verified in the code)

| Fact | Where | Consequence for "find more" |
|---|---|---|
| Every run is a brand-new campaign (new id, fresh `CampaignContext`, empty `query_log`). Nothing links batch 2 to batch 1. | `brain/cli.py:_run`, `brain/pipeline.py:run_campaign` | Batch 2 repeats batch 1's queries and finds the same top people. |
| The only repeat guard is `Store.blocked_reason`, and it blocks only results with status `sent/replied/followed_up/bounced`, matched **by email address**. | `brain/store.py:732`, `CONTACTED` at `:32` | People who were shortlisted, drafted, approved-not-yet-sent, or **rejected by the user** are not blocked: they come back in batch 2. |
| The remembered-prospects source feeds earlier campaigns' prospects back in, ranked first. | `brain/store.py:known_prospects`, `brain/channels.py:discover_memory` | Actively pushes repeats to the top. |
| Identity is only checked inside one run (`resolve`, `seen` in `backfill`). Org candidates with no address yet, a person found again under a different address, or an org later replaced by its decision-maker (new id) all slip through. | `brain/resolve.py`, `brain/pipeline.py:583` | Exact-email matching is not enough. |
| `max_per_org` / `diversify` caps per **run**, not per outreach. | `brain/rank.py`, `settings.max_per_org` | Batch 2 can contact a third person at the same company. |
| The review UI sends with no do-not-contact check; only the bridge export checks `blocked_reason`. | `ui/review_ui.py:~120`, `brain/bridge.py:export_to_review` | Two batches running at once, or an old pending draft, can still double-send. |
| Schema is `Base.metadata.create_all` with no migrations. | `brain/store.py:Store.__init__` | New **tables** are free; new columns on existing tables are not. Design below uses tables only. |

Two separate problems hide in "don't repeat people":

1. **Correctness** (never contact twice) is a ledger problem. It is cheap and must be exact.
2. **Supply** (keep finding *good* new people) is a discovery problem. A perfect ledger on a drained pool just
   returns empty batches, and every repeat that is found and then dropped late still costs search, fetch and LLM money.

## 2. Design

### 2.1 Outreach series (the thing that "find more" continues)

A **series** is one brief run in many batches. A batch is a normal campaign. New table `brain_series_members(series_id,
campaign_id, batch_no, created_at)`; the first campaign's id is the series id, so existing campaigns are batch 1 of their
own series with no migration.

- `brain run ...` starts a series implicitly. `brain more <series|campaign> -n 20` creates batch k+1:
  same `CampaignSpec` (sender, ask, topics, locations) with a new id and `target_count=n`.
- Series-level memory carries over by loading the previous batch's saved `CampaignContext` and merging: `query_log`
  (so `ctx.tried()` stops repeat queries), `tool_stats` (the bandit keeps what it learned), `rejected_reasons`, and
  seeds from outcomes (people who replied or were approved become `seed_names` for lookalike search).

### 2.2 Touch ledger: one source of truth for "have we been here"

New table `brain_touch`: `key` (PK), `contact_id`, `series_id`, `campaign_id`, `state`, `org_domain`, `updated_at`.

- **States**: `surfaced` (shortlisted to the user) → `drafted` → `approved` → `sent` → `replied` / `bounced` /
  `unsubscribed`; plus `rejected` (user said no, final), `standby` (user said "not now", parked), and `external`
  (imported do-not-contact / "already emailed" list, which the intake already asks for in `_UPLOAD_COMMON`).
  Progress states never move backwards; `rejected`, `standby` and the send states are explicit user/outcome events.
- **Hard vs soft** (this drives everything in 2.3 and 2.5):
  - **Hard-blocked, never returns:** `sent`, `replied`, `bounced`, `unsubscribed`, `spam`, `rejected`, `external`.
  - **Soft-blocked, a fallback pool:** `surfaced`/`drafted` that the user never decided on, and `standby`. They are
    excluded from normal batches and only come back through the recycle rule in 2.5.
- **The rejection signal lives in the wrong table today.** The review UI sets `rejected` only on the legacy `emails` row
  (`ui/review_ui.py:~160`); `bridge.sync_outcomes` syncs only sent/replied/bounced. P0 must extend it to sync
  `rejected` and `standby` into the ledger, and `brain more` must run that sync first, or a rejection never reaches the dedupe.
- **Keys** (one contact writes several rows, so any one match is a hit):
  - every candidate address, normalised (lowercase, strip `+tag`, Gmail dots), as `email:`
  - `orcid:`, `openalex:`, `github:`, profile URL host+path
  - `person:<name tokens>|<registrable domain>` (the formula already in `resolve.assign_id`)
  - `org:<registrable domain>` for organisations, and for any person at that org (feeds the per-org cap)
- **Fuzzy fallback** for records with no shared key: `names_compatible` + `orgs_compatible` from `brain/resolve.py`
  against ledger rows of the same org domain. The error costs are asymmetric: skipping a different person costs one
  prospect, emailing the same person twice costs trust, so the dedupe threshold may be looser than the merge threshold
  in `resolve` (initials and middle names allowed to differ).
- Transitions are written where the facts already change: `save_results` (surfaced/drafted), `Store.set_status`
  (approved/rejected), `record_outcome` and `bridge.sync_outcomes` (sent/replied/bounced). Writes are idempotent per
  `(campaign, contact)`, so resuming a crashed batch cannot double-count.

### 2.3 Exclusion policy (the user-facing option)

A single setting, `--no-repeat`, chosen at `brain run` / `brain more` and stored on the series:

| Mode | Blocks | Default for |
|---|---|---|
| `series` (default) | anyone surfaced, drafted, approved, sent, or rejected in this series, plus everything contacted anywhere in the workspace | bulk / "find more" |
| `workspace` | anyone in **any** campaign of this workspace in any state | users running several parallel briefs |
| `cooldown:<days>` | contacted/surfaced people come back only after N days with no reply and no bounce | re-engagement campaigns |

Rules that hold in every mode:
- `unsubscribed`, `bounced`, `spam` and suppression-list entries are permanent.
- `sent` is workspace-wide permanent (today's behaviour, now matched on all keys).
- **`rejected` never comes back** — not in the recycle pool, not after a cooldown. The only way back is the user's own
  explicit `brain release <contact>` (an undo for a mis-click, never automatic).
- **`standby` is "not now", not "no".** The user parks a prospect (a third button next to Approve / Reject in the review
  UI, and `brain standby <contact>`). A standby prospect is skipped by normal batches, keeps its existing draft, and
  returns only through the recycle rule in 2.5 or an explicit `brain recall`. Standby needs a new `EmailStatus` value in
  `orchestrator/state.py`; check that the legacy `status` column is a plain string so no schema change is needed.
- A per-org cap applies across the whole series (`max_per_org` counts earlier batches too; organisation-inbox
  campaigns cap at 1).

### 2.4 Enforce in three places (so repeats are cheap to avoid and impossible to send)

1. **Early, to save money.** Extend `Store.blocked_reason` to consult `brain_touch` (all keys, per the active mode).
   It is already called by `rank_stage` before any enrichment or judge spend, so a repeat dies at ranking, not after an
   LLM call. Also filter `known_prospects`, and seed `explore`'s `known = {_key(c) ...}` set (`brain/explore.py:151`) with
   the series' ledger keys so each tool's "new" count is honest and the budget allocator sees yield falling.
2. **Steer discovery.** Give the planner a few examples of what is already covered (never the whole list: it can be
   thousands) and rely on the `ctx.tried()` carry-over to avoid re-running queries.
3. **Final gate, as a safety net.** An atomic *claim* when a batch's shortlist is saved (insert-if-absent on the
   ledger key, inside the same transaction as `save_results`) so two concurrent batches cannot both take one person.
   Re-check at `export_to_review` (exists) and add the check to the approve/send path in `ui/review_ui.py`, which has none.

### 2.5 Fresh supply: keep finding good new people without lowering the bar

Dedup alone gives diminishing returns, so each batch widens along a **ladder**, only moving to the next rung when the
current one's *fresh yield* (new qualified candidates ÷ qualified candidates) drops below a threshold (start at 0.4):

1. New queries from the series' untried templates, then pages/cursors on sources that support them (Serper page,
   OpenAlex cursor, news date windows, HN month back). Needs an optional `cursor` in the adapter/hook contract
   (`brain/discovery/contracts.py`); check which sources can support it before promising this for all.
2. Lookalike / graph walk from the best earlier results and from anyone who replied (`walkers.py`, `graph.py` exist).
3. Widen geography (city → region → country), then neighbouring topics (the next tier of `expanded_terms`).
4. **Recycle (only when very few new matches *and* the user asked for more).** After rungs 1-3, if fresh results fall
   short (fewer than `max(3, n/2)` new), offer the **soft pool**: people surfaced earlier but never decided on, then
   `standby`, best score first. Never `rejected`, never anyone sent.
   - `--recycle ask|auto|never`, default `ask`: "Only 4 new found. 9 earlier suggestions are still undecided and 3 are on
     standby. Include them?" (`--yes` means `auto`).
   - Recycled people are **not re-judged or re-drafted**: their stored score and draft are reused (no LLM cost), and
     they are moved into the new batch, not copied. The legacy review tables key a draft by campaign
     (`stable_id(spec.id, c.id)` in `bridge.py`), so a copy would show the same person twice in the review UI.
   - They are labelled "previously shown" so the user knows why they are seeing them again.
5. Stop. Report: `Pool exhausted for this brief: 14 new of 20; 61 already covered. Widen location or topics to continue.`

The shortlist floor (`BRAIN_SHORTLIST_FLOOR`, 0.6) is **not** relaxed to fill a batch, and recycling does not lower it
either: only people who already passed it are eligible. A short, honest batch is the correct answer (consistent with
`coverage()` in `brain/pipeline.py`).

### 2.6 Bulk mode and guardrails

"Without any bound" is a product risk, not only a dedupe problem: unbounded sending burns the sender's mailbox.
- `brain more --until <total_sendable>` loops batches until the goal, the exhaustion rule, or a **budget cap**
  (search calls / LLM spend per series, extending `search_budget`) stops it.
- **Every email is human-approved, with no auto-approve** (decided). Bulk mode only produces drafts for review; the
  user approves or rejects each one. Sends still go through `email_dispatch/rate_limiter.py` (daily cap, min delay;
  counters should be persisted so the cap survives a restart). If auto-approve is ever wanted, it would be a later,
  separate decision limited to `sendable` drafts.
- **Bounce circuit breaker:** outcomes already record bounces; pause the series when the rolling bounce rate passes a
  threshold (suggest 5%) and say why.

### 2.7 What the user sees

- `Batch 3 of "sponsors for Advitiya": 17 new (0 repeats). Skipped before spend: 31 already surfaced, 6 already contacted, 4 org cap.`
- Funnel gains `dedup: {already_surfaced, already_contacted, rejected_earlier, org_cap, fresh_yield, rung}`.
- CLI: `brain more`, `brain seen <series>` (counts by state, who), `brain standby <contact>`, `brain recall <contact>`,
  `brain release <contact>` (undo a rejection), `brain run --exclude-file`.
- Review UI: Approve / **Standby** / Reject per draft, grouped by batch (note: the review bridge is not yet
  workspace-aware; see `docs/production-hardening.md`).

## 3. Phases

| Phase | Scope | Result | Size |
|---|---|---|---|
| **P0** | `brain_touch` + key normalisation + `blocked_reason` on all keys + writes from `save_results`/`set_status`/outcomes + **sync of rejected/standby from the review tables** + claim on save + approve-time check + `known_prospects` filter + tests | "Find more" stops returning anyone already surfaced, rejected or contacted. Works even if the user just re-runs `brain run`. | ~1-2 days |
| **P1** | Series table, `brain more`, context carry-over, explore known-set seeding, `--no-repeat` modes, org cap across batches, **Standby button + `standby`/`recall`/`release` commands**, funnel and CLI output | Batches are first-class and cheaper (repeats die before spend). | ~2-3 days |
| **P2** | Cursor support in adapters, the widening ladder, **recycle of the soft pool (`--recycle`)**, exhaustion report, `--until` with budget cap, lookalike from replies | Honest unbounded supply of new good people. | ~3-4 days |
| **P3** | Persisted daily cap and bounce breaker wired to the sender, review UI batch view, workspace-aware bridge | Safe to leave running. | ~2 days |

P0 alone fixes the bug the user described; P1-P2 make it efficient and sustainable.

## 4. Tests and measurement

- Invariant (hard fail): `repeat_rate == 0`. Across N consecutive batches of the same series no contact id, address,
  org-cap-violating domain, or fuzzy name+org match appears twice.
- Unit: key normalisation (`a.b+x@gmail.com` = `ab@gmail.com`); same person via different address; org inbox contacted then
  its decision-maker found; `Dr. S. Kumar` vs `Sanjay Kumar` at one org; `Sanjay Kumar` vs `Sunil Kumar` stay distinct;
  each mode, including cooldown expiry; `release`; **rejected is never recycled; standby and undecided are recycled only
  when fresh < `max(3, n/2)` and `--recycle` allows; a recycled person is moved (one review row, no second LLM call);
  a rejection made in the review UI reaches the ledger via the sync.**
- Pipeline (fake sources, like `tests/brain/test_pipeline.py`): batch 1 → batch 2 disjoint; kill and resume mid-batch
  leaves no phantom claims; two concurrent batches claim disjoint sets; rejected stays out; memory source filtered.
- Bench: add `fresh_yield@batch_k` and `cost_per_fresh_good` to `brain/bench.py` for a 3-batch scenario so the ladder is
  tuned on data, not feel. Quality guard: `good@K` of batch 3 must not fall below batch 1's by more than its Wilson CI.

## 5. Decisions

Decided (2026-11-20):
1. **Rejected never returns**, except by the user's own `brain release`. A separate **standby** state means "not now":
   parked, skipped by normal batches, recallable.
2. **Surfaced-but-undecided blocks people within the series**, but when new matches are very few and the user asked for
   more, those people (and standby) are offered again, best score first, without re-judging (section 2.5).
3. **Every email is human-approved**; no auto-approve for now.

Still open (recommended defaults in bold):
4. Per-org cap for people-targeting campaigns: **keep `max_per_org` but count across batches**; orgs-by-inbox: 1.
5. Default scope: **series** for `more`; plain `brain run` keeps today's behaviour plus P0's wider keys.
6. "Very few" threshold for recycling: **fewer than `max(3, n/2)` new**, adjustable.
