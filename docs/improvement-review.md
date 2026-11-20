# Code review: improvements by component

Date: 2026-11-20 · Branch: `discovery-v2`

Scope: read from the code, not run. `brain/pipeline.py`, `explore.py`, `intake.py`, `fit.py`, `rank.py`, `judge.py`,
`draft.py`, `llm.py`, `fetch.py`, `cache.py`, `enrich.py`, `searchplan.py`, `settings.py`, `store.py` (structure),
`bridge.py`, `discovery/router.py`, `discovery/precision/*`, `discovery/reach/*`, and the legacy send path were read.
`orgs.py`, `channels.py`, `tools.py` and the source manifests were only skimmed, so they get less detail.
Each item lists what was found, then what to change.

## 1. Repo hygiene

**Found**
- `outreach.db`, `HarshModi.pdf` (a CV), `run_logs/` (stored LLM inputs and outputs), the LaTeX build files
  (`main (3).*`), `hello.txt` and the root-level `test_*.py` files are all tracked in git. The remote is on GitHub.
- `outreach.db` shows as modified in every `git status`.
- There is no `.github/` folder, so no CI runs.
- `requirements.txt` pulls in langchain, chromadb, crawl4ai, playwright and sentence-transformers. The `brain/`
  code needs none of these.

**Improve**
- `git rm --cached` the database, CV and logs, and add them to `.gitignore`. If the repo is public, check what is
  already exposed.
- Split the requirements into `brain` and `legacy`.
- Add a CI workflow that runs `pytest tests/brain`.

## 2. Intake (`brain/intake.py`)

**Found**
- One LLM call fills a schema of about 30 fields (`_SCHEMA`), so one bad field costs a whole retry.
- `_ASK_VERB` (line 262) accepts almost any ask, because it includes words such as `list`, `role`, `write` and
  `position`. The "no clear action" check therefore rarely fires.
- `topic_stoplist` per playbook works around the model describing the sender instead of the target. It fixes the
  symptom, not the cause.
- The 30 intake eval cases were also used to tune the prompts, so a 30/30 score may not hold on new prompts.

**Improve**
- Use schema-constrained output, or split the call into "classify and target" and "details".
- Ask for a separate `target_description` field.
- Add held-out intake cases, written after the prompt is frozen.
- Replace the verb regex with a small LLM check, or a verb list tied to `ask_detail.type`.

## 3. Fit scoring (`brain/fit.py`)

**Found**
- `_term_hit` matches word prefixes, and phrases match when all their words appear in any order. `"sponsor"` hits
  `"sponsorship opportunities page"`. That gives false positives in "qualified".
- Support is a coarse formula, `0.45 + 0.2 * hits`. The qualified cut-off of 0.25 is an uncalibrated constant.
- `SIGNAL_KEYS` is computed at import time, so a user manifest loaded later cannot add signals.
- Anti-patterns are checked only against the name, title, bio and the first 3 evidence items.

**Improve**
- Use proximity or phrase matching, and downweight a hit that has no supporting sentence.
- Calibrate the threshold against judge verdicts.
- Make the signal registry lazy.
- Check anti-patterns against all the evidence.

## 4. Discovery (`explore.py`, `discovery/router.py`, `searchplan.py`)

**Found**
- The UCB bandit rewards "qualified per search call", and "qualified" comes from the weak rule in section 3. Until a
  learned model exists (50 or more examples per playbook), the bandit optimises for the loose definition. The
  `final_live` run is consistent with this: good@K 23/52 against 31/52 with hand-written specs.
- `SearchPlanner.variants()` only swaps a topic or place string, so it adds little diversity. Lookalike queries use
  the generic "competitors" and "alternatives" templates.
- The budget split is fixed: 55% discovery, about 30% contact details, 15% backfill. `search_budget` is
  `10 + 3 * target`, capped at 80.
- Tavily is configured, but it only works as a fallback when there is no Serper key.

**Improve**
- Use pre-judge scores as the bandit reward during a run. They are cheap and arrive immediately.
- Move budget between stages based on observed reachability, for example more on enrichment when the reachable
  share is low.
- Add source-specific query generation, such as queries for proof pages, in place of string swaps.
- Run Serper and Tavily on a sample to compare yield.

## 5. Filtering and ranking (`brain/rank.py`)

**Found**
- `in_region` (line 52) uses plain substring matching (`t in where`). A short term such as `goa` matches "goals".
- BM25 inverse document frequency depends on the current candidate pool, and scores are normalised by the best
  document, so pre-scores are not comparable across rounds.
- `embedding_scores` reloads the SentenceTransformer on every call.
- The pre-score weights in `DEFAULT_WEIGHTS` are hand-set constants.

**Improve**
- Use word-boundary regexes for region terms.
- Fix the IDF at the start of a rank pass, or use a stable score.
- Cache the embedding model.
- Fit the weights from labelled data once it exists.

## 6. Precision layer (`discovery/precision/`)

**Found**
- `final_live` showed the pre-judge being over-strict: it rejected 41 of 65 sponsor candidates, partly because the
  intake topics described the sender and not the target.
- Fit-model training data is biased. Only top-ranked candidates reach the judge, and the pre-judge adds only
  negatives below 0.4. The model learns to imitate the LLM judge, not the truth.
- Training is hand-written gradient descent with no holdout, so nothing shows whether the learned model beats the
  rule.

**Improve**
- Log pre-judge versus judge agreement on every run, with false-reject rate as a headline metric.
- Cross-validate before saving a model, and fall back to the rule unless the model clearly beats it.
- Weight examples to correct the selection bias.

## 7. Enrichment and email reach (`enrich.py`, `discovery/reach/`, `emailfinder.py`)

**Found**
- Obfuscated emails such as `[at]` and `(at)` are decoded. Cloudflare `data-cfemail` is not decoded; I found no
  handling for it.
- The prior pattern `first@company` has confidence 0.5 and is checked only for MX. MX proves the domain accepts
  mail, not that the mailbox exists. A wrong guess bounces and harms the sender's reputation.
- A `bounced` status exists in the store, but nothing feeds bounces back into pattern confidence.
- JavaScript-rendered team pages are not read by `brain/`.

**Improve**
- Decode `data-cfemail`, and read schema.org `Person` JSON-LD for contact fields.
- Learn from bounces by downweighting patterns that bounce.
- Add catch-all detection and optional SMTP probing, or use Hunter, for guessed addresses.
- Add an opt-in headless fetch for pages that return little content.

## 8. Judge (`brain/judge.py`)

**Found**
- The final score mixes the model score, criteria scores, `0.2 * prescore` (lexical noise) and a reach factor. The
  0.7 "good" threshold is therefore an uncalibrated blend.
- The judge sees at most 5 evidence items of 180 characters each. Batches of 8 can show position bias.
- No repeat run measures noise.

**Improve**
- Drop the prescore term from the final score.
- Judge fit and reachability separately and combine them afterwards.
- Calibrate scores against human labels, for example with isotonic regression.
- Re-judge borderline cases (0.6 to 0.8) with the order shuffled.

## 9. Drafting (`brain/draft.py`)

**Found**
- The deterministic checks are the strongest part of the repo: grounding, quoted titles, numbers, conflations, one
  ask and spam words.
- The fact-check uses the same model family as the writer, so their errors correlate.
- `salutation` takes `tokens[-1]` as the surname, which is wrong for "van der Berg" and for single-name people.
- The banned-phrase list is English-only.
- Each draft gets one retry.

**Improve**
- Run the fact-check on a different model.
- Add a name-particle table for surnames.
- Extend the banned-phrase list by language.
- Show a "why this person" line per draft in review.

## 10. Compliance and safety

**Found**
- The opt-out line is added only for commercial playbooks, and a postal address is optional. There is no
  `List-Unsubscribe` header.
- Suppression is checked during discovery. I did not find a re-check at send time.
- There is no handling of EU or India rules (GDPR, DPDP) for bulk cold email to individuals.

**Improve**
- Re-check suppression at send time.
- Require a postal address for commercial sends.
- Add `List-Unsubscribe` headers.
- Flag EU and India recipients for a compliance note, or block them by default.

## 11. LLM layer (`brain/llm.py`)

**Found**
- The "one repair attempt" loop (line 181) re-sends the same request at temperature 0, up to 6 times. A failed parse
  will probably fail the same way each time. It should feed the parse error back to the model.
- `PRICES` has no entry for Groq models, so `est_cost_usd` reports 0 for them.
- `self._reasoning` is set per call on the shared object, which would be unsafe if calls ran in threads.
- Drafts are cached at temperature 0.5, so a re-run returns the same draft and one cannot be resampled.

**Improve**
- On a parse failure, send the error back and ask for corrected JSON.
- Add Groq and Claude prices.
- Pass `reasoning` as an argument.
- Add a `no_cache` option for re-drafting.
- Use JSON-schema output where the provider supports it.

## 12. Fetch and cache (`brain/fetch.py`, `brain/cache.py`)

**Found**
- Full HTML (up to 2 MB per page) is stored in SQLite. The cache commits on every write, never evicts expired rows
  and does not use WAL mode.
- The user agent `OutreachResearchBot/0.1` has no contact URL, which is the usual courtesy for crawlers.

**Improve**
- Store compressed or extracted text.
- Enable WAL and batch commits.
- Add an expiry sweep.
- Put a contact or info URL in the user agent.

## 13. Store and memory (`brain/store.py`)

**Found**
- Memory, patterns and priors are shared across every campaign in one database. The README says this is not safe
  for multiple users.
- No migrations are used (alembic is in the requirements, but I did not see it used).

**Improve**
- Add an account or workspace ID to every memory table before any multi-user deployment.
- Use real migrations.

## 14. Sending and the legacy side

**Found**
- `brain` has no send path. `bridge.py` copies drafts into the legacy `professors` and `emails` tables, and the
  Streamlit UI sends through `agents/gmail_sender.py`. Two schemas and two status machines are kept in sync by hand.
- The rate limiter keeps its counters in memory, and its own docstring says they should be in the database. A restart
  resets the daily and weekly limits.

**Improve**
- Build a native review-and-send step on `ResultRow`, with persisted limits, bounce and reply handling, and the
  suppression check.
- Retire the bridge, and then the LangGraph legacy path.

## 15. Evaluation (`brain/bench.py`, `evals/`)

**Found**
- Good@K is scored by an LLM judge of the same kind that ranks the candidates, so the metric is partly circular.
  Label files exist, but the v3 plan says the labels are still unfilled.
- Grader noise was measured at about ±1 email per scenario, and live runs cost real money, so a difference of 23
  versus 31 good@K on one run is hard to interpret.

**Improve**
- Fill the human labels, even 30 per scenario.
- Hold out scenarios that nothing is tuned on.
- Report confidence intervals.
- Run each live scenario twice before drawing conclusions.
- Re-measure the intake fixes (`86ced90`, `606650d`) live; they have not been measured yet.

## 16. Tests

**Found:** about 140 offline tests, which is a good base. There is no CI, and no contract test that notices when a
live source (arXiv, NSF, job-board APIs) changes its response shape.

**Improve:** add CI, and a periodic smoke test of each source against a recorded schema.

## Suggested order

1. **Repo hygiene and CI** (section 1). It is cheap and removes a possible privacy problem.
2. **Human labels and held-out scenarios** (section 15). Without them, the discovery tuning in sections 3, 4, 6 and
   8 cannot be measured properly.
3. **Judge calibration and fit-score fixes** (sections 3 and 8). These change ranking quality directly.
4. **Send path and compliance** (sections 10 and 14). They are required before real deployment.
5. **LLM parse-retry and region matching** (sections 11 and 5). They are small, concrete bugs.
