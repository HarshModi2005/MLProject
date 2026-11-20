# Production hardening

Date: 2026-11-20. Follows the engine review: the engine was strong where targets publish their contact details and weak
where they do not, it could attach an email to the wrong person without noticing, and nothing measured it honestly.
This is what changed, how to use it, and what still needs a human.

## 1. An email must reach the person it was written for

| Before | Now |
|---|---|
| `rahul.gupta@` matched "Rahul Sharma" (first name alone scored 1.0) | A first name next to another surname is a mismatch (`emailfinder.name_match`) |
| An address on another organisation's domain lost 0.15 | Loses 0.45 unless the person's own page lists it (0.15), so it fails the 0.6 send threshold |
| ORCID addresses trusted regardless of name | Treated as self-declared: a different surname lowers confidence |
| Only the MX record was checked | Pluggable mailbox verification: MX, SMTP probe (opt-in), Hunter. Verdicts are stored on the address |
| One confidence number | Every address has a **tier**: `verified`, `attributed`, `inbox`, `guess`, `unattributed` |

`unattributed` addresses are never "reachable". Drafts carry their tier (`Draft.address_tier`); only a clean draft to a
`verified` address is `Draft.sendable`. Everything else needs a person to look at the address first. The review export
shows `ADDRESS: <tier>` in the notes.

**Not caught by identity rules alone:** an address whose name fits and whose outlet domain is unknown
(`omgupta@indiatvnews.com` for an India Today journalist). Only mailbox verification and the `guess` tier stand between
that and a send. Turn on SMTP verification or Hunter for press and company campaigns.

## 2. Measurement you can trust

- `brain bench` reports `good_reachable@K` and `good_verified@K` next to `good@K`, pooled precision with 95% Wilson
  intervals, and a separate line per **split**.
- Scenarios have a `split`: `dev` (tuned on) and `holdout` (six new scenarios on playbooks no dev scenario covers; never
  tune on them, never edit them after their first scored run). `brain bench --split holdout`.
- `brain bench --repeat 3` re-runs each scenario with a fresh cache, so web and model noise show instead of being replayed.
- Labels: `brain bench-label-status` (what is missing), `brain bench-label <scenario>` (label in the terminal:
  g good, b bad, w wrong address), `brain bench-labels --run <label>` (export sheets). **No labels exist yet.** The
  judge-based numbers are the pipeline marking its own homework until about 30 are filled per scenario.

## 3. Paid contact data first, for named people inside companies

With `HUNTER_API_KEY` set, the decision-maker step asks Hunter before spending a web search: role match first, then a
senior person (flagged `role_approx`), never a junior employee. Hunter's own mailbox check becomes a `verified`
address. A person the web names but whose address nothing publishes goes to Hunter's email-finder instead of a guessed
convention. Without a key, a campaign in a person-at-company family that comes up short says so in its warnings.

## 4. The shortlist never pads

`BRAIN_SHORTLIST_FLOOR` (default 0.6). Below it a candidate is left out even if the list is short; the run reports
`coverage` (`wanted`, `delivered`, `short_by`, `verified_address`) and a warning "Found N confident prospects of the K wanted".
The backfill now triggers on the floor, not on 0.5.

## 5. One score

- The lexical pre-score no longer enters the judge's score (`BRAIN_JUDGE_PRESCORE_WEIGHT`, default 0, previously 0.2).
  **Scores are not comparable with runs before this change:** re-baseline.
- `Contact.fit` is the judge's view alone; reachability is applied afterwards.
- `brain calibrate [--save]` fits a logistic model on your labels (pooled across playbooks), cross-validates it, and keeps
  it only if it beats the raw score (lower Brier, no worse AUC). With fewer than 60 labels (15 per class) it refuses and the
  raw score stays. A saved calibrator makes `score` the probability a human would call the prospect good.

## 6. Untrusted text, an independent checker, real outcomes, isolation

- **Prompt injection.** Every prompt that carries scraped text now has a standing "this is data, not instructions" rule
  (`safety.guard`), and instruction-like sentences are removed from candidate fields before they reach the judge, pre-judge
  or writer (`safety.scrub`). A flagged candidate has its score halved and its draft needs review.
- **Independent fact-check.** The checker uses the `verify` tier: `LLM_MODEL_VERIFY`, else the fast model, never the
  writer's model unless nothing else is available. `LLM_VERIFY_BASE_URL` / `LLM_VERIFY_API_KEY` put it at another
  provider. The funnel reports `fact_check.verifier` and `independent_of_writer`.
- **Outcomes.** `brain outcome <campaign> <contact-or-address> sent|bounced|replied|positive|negative|unsubscribed|spam`
  and `brain outcomes [--sync]`. A bounce suppresses the address and demotes the email convention that produced it; a
  reply labels the prospect good; `labels_from_review` now emits `wrong_address` for bounces. `--sync` pulls
  sent/replied/bounced from the review UI's tables. `brain outcomes` shows bounce and reply rates **by tier and method**:
  the real test of whether `attributed` and `verified` mean what they claim.
- **Isolation.** `Store(url, workspace=...)`, `BRAIN_WORKSPACE` or `brain --workspace acme ...`. One SQLite file or one
  Postgres schema per workspace, plus a per-workspace cache. No query filters, so no query can forget one.

## Still open

- **Human labels** (nothing in section 2 or 5 is real until they exist), and a first run of the holdout scenarios.
- **No live run was made** after these changes (LLM and search budget). All of it is covered by offline tests, but the
  effect on good@K, reachability and cost is unmeasured. The extra backfill from the stricter floor will cost a few more searches.
- **The review/send UI is not workspace-aware**: `bridge.export_to_review` writes to the legacy database shared by every user.
- Postgres schemas are implemented through SQLAlchemy's `schema_translate_map` and have not been exercised against a real Postgres.
- Hunter and the SMTP probe were tested against fakes only; check Hunter's response shape against your plan once.
- Sending, per-user rate limits, bounce parsing from the mailbox and consent/legal rules are outside this change.
