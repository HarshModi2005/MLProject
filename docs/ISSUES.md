# Issues found by an independent LLM judge

Date: 2026-11-20. Tool: `tools/llm_judge.py`. Raw output: `evals/results/llm_judge_v3_live.json`.

## How this was measured, and what it can't tell you

- **Judged run:** `v3_live` (live, 17 Nov), read from `evals/results/run_v3_live/*.db` and the emails in
  `evals/results/drafts_v3_live.json`. No discovery spend; the judge cost about $0.10 (16 calls).
- **Judge:** OpenAI `gpt-4.1`. The pipeline's judge, writer and checker run on Groq `gpt-oss`, so this is a different
  model from a different vendor. It saw what the writer saw (evidence, sender facts, address and its tier).
- **Sample:** 9 scenarios, 46 shortlisted prospects, 27 drafted emails (the top 3 per scenario).
  `speakers_ai_summit` shortlisted nothing, so it has no prospects to judge.
- **Not ground truth.** The judge has no web access, so it can't check whether an address works or whether a named
  person really holds the role. There are still no human labels (`evals/labels/*.json` are empty sheets).
- **This run predates the 20 Nov hardening** (address tiers, 0.6 floor, no padding). Re-run the judge on a fresh run
  before treating any item here as still open. Items I-03 and I-06 in particular may already be partly fixed.
- **The judge itself was too lenient on emails.** It gave every email grounded ≥4, and said "would send" for 26 of 27.
  Reading the drafts, I found defects it approved (I-01, I-03, I-04). Those are now caught by
  deterministic checks in the tool (`lint_email`), reported separately. Treat the judge as a triage signal,
  not a send gate.

## Scorecard

| Scenario | n | good | borderline | bad | mean fit /5 | emails "would send" |
|---|---|---|---|---|---|---|
| academic_gnn | 5 | 2 | 3 | 0 | 4.2 | 3/3 |
| b2b_app_devs | 4 | 4 | 0 | 0 | 4.5 | 3/3 |
| b2b_dental_bangalore | 6 | 5 | 1 | 0 | 4.8 | 3/3 |
| distributors_solar | 5 | 4 | 0 | 1 | 4.0 | 3/3 |
| grants_csr_india | 5 | 3 | 2 | 0 | 4.4 | 3/3 |
| investors_edtech | 5 | 3 | 1 | 1 | 3.6 | 3/3 |
| jobs_ml_startups | 5 | 3 | 1 | 1 | 4.0 | 2/3 |
| pr_ai_media | 5 | 4 | 1 | 0 | 4.4 | 3/3 |
| sponsors_tech_fest | 6 | 4 | 2 | 0 | 3.8 | 3/3 |
| speakers_ai_summit | 0 | – | – | – | – | – |
| **Total** | **46** | **32 (70%)** | **11 (24%)** | **3 (7%)** | | **26/27** |

Cuts that matter:

| Cut | good | borderline | bad |
|---|---|---|---|
| Organisation targets (shared inbox), n=25 | 22 (88%) | 2 | 1 |
| Named-person targets, n=21 | 10 (48%) | 9 | 2 |
| Address tier `inbox`, n=25 | 22 | 2 | 1 |
| Address tier `guess`, n=10 | 7 | 1 | 2 |
| Address tier `attributed`, n=9 | 3 | 6 | 0 |

Pipeline score vs judge verdict: mean pipeline score is 0.743 for judged-good, 0.704 for borderline and 0.736 for
bad. **The score does not separate bad prospects from good ones** (only 3 bad, so treat as a weak signal).
The three bad ones ranked #3, #2 and #3 in their lists (Voltah 0.787, Mark Grovic 0.737, Kai Brandt 0.685).

Deterministic email lint over the 27 drafts: `pipeline_meta_in_body` 6, `raw_address_in_body` 2, `junk_greeting` 2
(plus a page heading shown as a person in 3 shortlisted prospects, I-02). `draft.check` reported zero issues on all of them.

---

## P0: would send a wrong or embarrassing email

### I-01 Decision-maker lookup attaches the wrong or an unverifiable person
**Status: partly fixed (20 Nov).** A decision maker whose name fails `looks_like_person_name` ("At SIM") is no longer greeted; the draft falls back to "Hello <Org> team,". Still open: corroborating the person (org + role in the snippet, no social-post sources, `dm_confidence`).
`brain/orgs.py:363` `_find_decision_maker` → `people_from_serp`; greeting built in `brain/draft.py:90` `salutation`.
Six of the 27 emails greet a named "decision maker" found by a web search. Evidence from the stored records:

| Org | Greeted as | Where the name came from | Problem |
|---|---|---|---|
| E-Class | "Hi At," | `At SIM`, "Program Officer", an Instagram reel | Not a name |
| Photovoltaic Solar (Gujarat) | "Hi Delgado," | `Delgado Javier Javier`, "Director, Photovoltaic Engineer", a LinkedIn profile | Surname used as first name; no sign the person works there |
| Voltah (Gujarat) | "Hi Helen," | `Helen Pappas`, "Owner", a LinkedIn profile | Unverified; implausible owner of a Gujarat manufacturer |
| Dr Braces | "Hi Afsaneh," | `Afsaneh Rangiani`, "Owner", an Instagram post | The record's own evidence says the clinic is "Led by Dr. Salman Khan" |
| Ajuniorvc | "Hi Aviral," | `Aviral Bhatnagar`, "Partner", LinkedIn | Plausible |
| Gracerenew | "Hi Manish," | not checked | Unknown |

Why it matters: a greeting by name raises reply odds and also makes a wrong name obvious to the reader.
The judge approved all of these (it saw the `decision maker` line and did not question it).
Fix: accept a decision maker only if the search snippet names the organisation and the role together, reject social-reel
and post URLs as a source, require a two-token name with each token ≥3 letters that is not a preposition, and when
unsure fall back to `Hello <Org> team,` instead of a person. Record a `dm_confidence` and make `salutation` honour it.
Check: `junk_greeting` and `person_greeting_on_role_inbox` lint counts go to 0 on a rerun, and 20 hand-checked
decision makers are right ≥90% of the time.

### I-02 Page headings are shortlisted as people
**Status: fixed (20 Nov), pending a live rerun.** `looks_like_person_name` now rejects the heading words involved (corporate, governance, frontline, workplace, investor, relations…) and an acronym next to ordinary words ("At SIM"); the existing `plausible_person` gate in `brain/pipeline.py` uses it. Customer-service addresses (`frontline-service@`, `support.*@`, `services@`) are dropped in `Contact.add_email` (`brain/models.py`). The pipeline already rejected "Partner Spotlight" before this change, so the v3 run predates part of it. Tests: `tests/brain/test_cheap_fixes.py`.
Three shortlisted prospects have a section heading as their name and `kind=person`:
`Corporate Governance` (Bharat Forge, 0.67), `Partner Spotlight` (Databricks, 0.65), `Frontline Workplace` (TeamViewer, 0.64).
Their addresses are `corporate.communication@`, `partners@` and `frontline-service@` (a support inbox) for a
sponsorship ask. The judge marked two of them borderline and one good ("Partner Spotlight … plausible partner contact").
Fix: a name-shape gate where candidates are created (reject names made of heading words such as governance, spotlight,
workplace, careers, newsroom, overview); when the page is about an organisation, create `kind=org` with the page's
organisation as the name. Add a role-fit check on the address local part (support/service/noreply are never a fit for
sponsorship, partnerships, press or fundraising asks).
Check: `page_heading_as_person` lint is 0 on the shortlist, not just on the emailed top 3.

### I-03 Pipeline internals appear in the email body
**Status: fixed (20 Nov), pending a live rerun.** Cause: the writer prompt handed the model the inbox address (`brain/draft.py`, the RECIPIENT line). The address is gone from the prompt, the system rules forbid mentioning address/inbox/"reaches", and `draft.check` now fails any draft that shows an address or talks about an inbox (`leaks()`), which triggers the existing retry. Replaying the 27 judged drafts through `leaks()` flags all 6 that had the problem and none of the other 21.
6 of 27 emails say things like "I am writing to the team inbox at Voltah, hoping this message reaches you."
Two paste the raw address into the text: "Writing to drbracesandfaces@gmail.com hoping this reaches your team."
`draft.check` (`brain/draft.py:195`) found no issue, and the judge scored them 5/5 send.
Fix: add hard checks in `check`: no raw address in the body, no "inbox", no "hoping this reaches". Fix the prompt
or the writer's template that produces them (the inbox variant of the ask, `Ask.inbox_text`).
Check: `pipeline_meta_in_body` and `raw_address_in_body` are 0 across all campaigns, as a unit test on `check`.

### I-04 Competitors and wrong org types get shortlisted for B2B asks
`distributors_solar`: Voltah is a solar inverter manufacturer, and the sender is also a manufacturer of inverters
that wants distributors. Voltah scored 0.787 (rank 3 of 5), got a draft that says "Voltah is known for manufacturing
high-performance solar inverters" and asks them to distribute the sender's inverters. The judge flagged the claim as
unsupported but still said "would send".
Same pattern: `jobs_ml_startups`: Topstartups (a startup *directory*, 0.685, rank 3), whose email claims "I noticed your AI startup
listings include a Computer Vision Engineer … role" (judge: don't send; correct).
Fix: make "role in the value chain" a hard criterion for partnership and distribution asks, and derive competitors from the
sender's own offer (same product ⇒ competitor). Classify aggregators and directories as non-targets for hiring and
partnership asks (`org_class` exists in flags; use it as a gate, not just a flag).
Check: on a rerun, no manufacturer in the distributor list and no directory in the job list.

### I-05 Wrong geography or wrong kind of investor/journalist ranks high
`investors_edtech`: Mark Grovic (US firm, `guess` tier, 0.737, rank 2); Christopher Nyren (unclear India evidence).
`pr_ai_media`: Charlotte Lang, Computer Weekly (UK outlet) for an India launch.
The campaign's stated location is a soft signal in ranking, not a limit. Combined with the finding above that the
score does not separate bad from good, a wrong-geography person can outrank the right ones.
Fix: apply stated geography and org type as caps after judging (score × 0.5 or drop) rather than letting them average in.
Check: judge `wrong_geography` count 0 on `investors_edtech` and `pr_ai_media`.

---

## P1: coverage and quality gaps

### I-06 A scenario returns nothing, silently
`speakers_ai_summit`: 350 unique candidates, 15 qualified, 2 reachable, 0 passed the judge, 0 shortlisted, 16 searches,
226 s, `warnings: []`. 313 of the 350 came from OpenAlex and arXiv (academic sources), and the `events` source's 37 hits
qualified none. For a speakers campaign the useful sources (conference speaker pages, podcast guest lists) did not produce.
Fix: improve qualification of `events` results, add speaker-list adapters, and always warn "Found 0 of 5" so the zero is visible.
The 20 Nov hardening adds `coverage` and a warning, so first check whether a rerun already reports it.

### I-07 Person-targeted campaigns are the weak half
Persons are good 48% of the time against 88% for organisations. Judge reasons in `academic_gnn` and `b2b_dental_bangalore`:
no title on 3 of 5 professors ("No title listed; likely faculty but not certain"), and `bhaskarjyoti01@gmail.com` for
a professor was tiered `attributed` although it is a personal Gmail address (judge: "doubtful"). `Dentistree` and several
clinics use Gmail as a business inbox, tiered as `attributed` or `inbox`.
Related: `smishra@niser.ac.in` is greeted "Dear Professor Mishra" while the record has no title; the title is
assumed by the playbook, not read from evidence.
Fix: pull titles from the institution's staff page before shortlisting; cap the tier at `guess` for free-mail addresses
on a person target; use "Dear Dr/Prof." only with a title in the evidence, else the plain name.

### I-08 Organisation targets are reached through generic inboxes
In 6 of 9 lists the judge's top systemic issue was "no named decision-makers / mostly generic inboxes", and it expected
medium reply odds for those emails. The lever is I-01 (finding the right person reliably), not finding more people.
A wrongly named person (I-01) is worse than an honest "Hello X team".

### I-09 Lists lack variety the ask needs
Judge `missing` notes: `pr_ai_media` has no podcast hosts though the persona asks for "journalists and podcast hosts"
(a podcasts source exists but did not contribute); `investors_edtech` no angel networks; `jobs_ml_startups` no CTO/engineering heads
(two founders, a directory's founder, and `Valiance Solutions` twice: once as the org inbox and once as its CEO);
`grants_csr_india` no large-company CSR heads; `sponsors_tech_fest` includes Reliance, judged borderline (not a developer or edtech brand).
Fix: enforce one entry per organisation at the final cut (the duplicate shows `max_per_org` is not applied across
org-inbox and person entries); check that each persona role named in the spec appears in the top K, and backfill
from the matching source if one is missing.

### I-10 The two LLM graders share blind spots
The pipeline's own email grader and this judge disagree on 2 of 27 (`krishangtechnolab`, `charlotte.lang` are "no" to the
pipeline and "yes" to the judge). Both approved the defects in I-01, I-03 and I-04. Different vendors did not remove it.
Fix: keep deterministic checks as the gate (see `lint_email`), run an adversarial "find the flaw" pass for the
borderline cases, and calibrate both graders against human labels before using either as a metric.

---

## P2: smaller things

- I-11 `Hello Excel Dental Care | Jayanagar, Bengaluru team,`: already fixed by `team_name` before this review (it returns "Excel Dental Care" today); the v3 drafts predate it.
- I-12 Bench senders are placeholders ("Test Founder"), so no real sender facts are exercised. Judge a run that uses a real profile before launch.
- I-13 Only the top 3 per scenario are drafted, so the other 2–3 prospects' emails are never judged.
- I-14 Intake evaluation is 30/30 on cases the author wrote; it can't show intake is good on unseen requests. Add cases from real users.

## What the judge says is working

- Org-inbox targeting for local businesses is strong: dental 5/6 good, app devs 4/4, fit 4.5–4.8.
- Drafts are mostly grounded: only two `unsupported_claim` flags in 27 (Voltah, Topstartups); personalisation 4–5.
- Academic top picks are right (two professors with direct GNN work), and PR picks are on topic (4/5 good).
- Identity checks hold: the judge found one doubtful address in 46 (a Gmail on a professor), none clearly wrong.

## Suggested order

1. ~~I-03 and I-02~~ done (20 Nov); confirm on the next live run.
2. I-01: validate decision makers or fall back to "Hello X team".
3. I-04 and I-05: hard gates for org type and geography.
4. Re-run `tools/llm_judge.py` on a fresh post-hardening bench run; compare the scorecard above.
5. Label ~30 candidates per scenario (`brain bench-label`) and measure how often this judge agrees with you. Until then every
   number here is the judge's opinion.

Reproduce: `python tools/llm_judge.py --run <label> --drafts evals/results/drafts_<label>.json` (needs `OPENAI_API_KEY`;
`JUDGE_MODEL` / `JUDGE_BASE_URL` / `JUDGE_API_KEY` change the judge; results are cached, so a rerun is free).


---

# Round 2: stress cases and scale (20 Nov, after the fixes above)

New scenarios in `evals/scenarios/stress_*.json` (`--split stress`): 12 harder cases (multi-constraint investors, exclusion
lists, Spanish outreach, cross-border distributors, a senior hire, PhD positions, hyper-local, podcasts, open-source
maintainers, municipal IoT) and two large ones (50 investors, 40 clinics). Run with `brain bench --split stress --draft`.
Judge: `python tools/llm_judge.py --run <label>` (it now reads the drafts the run stored in its own databases).
**Only part of the set finished** (see "Not run" below); what finished is reported here.

## Scale and infrastructure findings

- **S-01 The default LLM provider cannot be used at scale.** On Groq's on-demand tier the pipeline models hit 8,000 tokens per minute
  (`gpt-oss-120b`/`gpt-oss-20b`) and 1,000 output tokens per minute (the `qwen` fallback). Two benchmark processes sharing one
  key throttled each other to a crawl, and org-page extraction, pre-judge and judge batches failed with HTTP 429
  (some candidates are then silently left out as "unjudged"). I switched the stress runs to OpenAI (`LLM_PROVIDER=openai`).
  Fix: a per-provider token-bucket in `brain/llm.py` (wait instead of fail), a warning when more than a few batches are lost
  to rate limits, and a documented minimum tier for production.
- **S-02 Wall-clock is dominated by fetching, not the model.** Each scenario takes 10-20 minutes with 30-60 searches; ten scenarios
  in sequence is hours. Bench runs scenarios one after another, so a failure late in the set loses the summary JSON
  (it is only written at the end). Fix: write the report after every scenario, add `--parallel N` for scenarios.
- **S-03 Scale numbers (OpenAI models, one process each, 20 Nov):**

  | Run | wanted | shortlisted | reachable | searches | LLM calls | $ | wall-clock |
  |---|---|---|---|---|---|---|---|
  | stress_scale_clinics_40 | 40 | 31 | 64 | 61 | 80 | 0.068 | 8.7 min |
  | stress_scale_vc_50 | 50 | 31 | 37 | 133 | 168 | 0.080 | 17 min |
  | stress_germany_machinery | 12 | 7 | 13 | 31 | 51 | 0.030 | n/a |
  | stress_investors_multi | 15 | 7 | 11 | 44 | 64 | n/a | n/a |

  Cost stays under 10 cents per run; time and searches grow roughly linearly with the target. **Neither scale run filled its quota**:
  31 of 40 and 31 of 50. The 0.6 floor stops padding and the coverage warning says so, which is the intended behaviour, but it means "50 investors in India" is
  really about 30 for this engine today. In the VC run, 204 of 227 candidates qualified and only 37 had a confident address,
  so reachability, not discovery, is the limit (the run's own warning asks for `HUNTER_API_KEY`).

## Quality findings (independent judge)

| Scenario | n | judge good / borderline / bad | emails "would send" |
|---|---|---|---|
| stress_scale_clinics_40 | 40 | 36 / 3 / 1 | 12/12 (judge's view; lint below disagrees) |
| stress_germany_machinery | 7 | 2 / 4 / 1 | 3/7 |
| stress_scale_vc_50 | 31 | 21 / 7 / 3 | 12/12 (first 12 judged) |
| stress_investors_multi | 7 | 3 / 4 / 0 | 3/7 |

### J-01 Organisations greeted by a made-up first name through a shared inbox (fixed)
`info@omhappyteeth.com` got "Hi OM,", `info@clinicaderm.in` "Hi Clinica,", `contact@jehangiroracare.com` "Hi Jehangir,",
`info@sorma.net` "Hi Sorma,". The "one-person business under its owner's name" rule fired for any name that looked like
two or three capitalised words. The judge approved four of these. Fix: never greet an organisation by a person's name when its address is a shared inbox
(`brain/draft.py` `salutation`). Test: `test_org_named_like_a_person_is_not_greeted_through_a_shared_inbox`.

### J-02 For distribution asks the shared-inbox rule replaced the real ask with a referral request (fixed)
`stress_germany_machinery`: three of four drafts were judged `ask_misfit`/`ask_unclear`: "Directly ask if they are interested in
distributing your products, not for referrals." The `org_inbox` situation told the writer the ONE ask is the shared-inbox
ask ("point me to whoever handles this") when one is set. Now the main ask is made in full and the referral line may only be a closing clause.

### J-03 A website builder is shortlisted as a clinic (fixed)
`Weebly` (`weebly.com`) with a clinic's Gmail address, judged bad. Site builders and free hosts (weebly, wixsite, wordpress,
blogspot, business.site, godaddysites, squarespace, linktr.ee …) are now in `AGGREGATOR_DOMAINS` and skipped before judging.

### J-04 Multi-country requests cover only some countries (open)
`stress_germany_machinery` asked for Germany, Italy and Poland: the shortlist has Italy and Poland and no German company
(judge: "No German targets"). `stress_scale_clinics_40` asked for Mumbai, Pune and Hyderabad: judge notes "No Hyderabad clinics" in one batch and
"No Pune clinics" in another. Fix: per-location quotas in the search plan and a coverage line per requested location
(`coverage` in the funnel reports totals only).

### J-05 Manufacturers and ambiguous organisations still pass distributor asks (open)
Germany run: UOP S.p.A. (a manufacturer) was shortlisted; four more are "unclear if distributor or manufacturer"
(Luna Polska, POLTRA, Sorma, Baritools). The competitor gate (I-04) depends on the judge model knowing; evidence of the
organisation's role (does the site say "distributor", "dealer", "reseller", "authorised") is not extracted. Fix: add a
`role_in_value_chain` signal from the organisation's own pages and make it a required criterion for distributor/reseller playbooks.

### J-06 Personal Gmail addresses for local clinics tier as `guess` and `inbox` (note)
Many clinic contacts are a doctor's own Gmail (`drrinkykapoor@gmail.com`). That is plausible for solo practices
(judge: good), but is the case where "reachable" and "verified" differ most. Keep these out of auto-send until bounce rates by tier are known (`brain outcomes`).

### J-07 New kinds of directory slip past the aggregator list (open)
`stress_scale_vc_50`: Startup Grants India (0.84, rank 1 by score), PitchDecks.in (0.74) and startupfunds.in (0.69) are directories or
aggregators, not funds; the pipeline's model did not set `competitor_or_directory`. The static list and name patterns cannot know new sites. Fix: a second cheap
check on the organisation's own homepage text ("we list/connect/curate …" vs "we invest …"), or a required criterion "makes the investment itself".

### J-08 The writer claims a fit the evidence does not state (open)
`stress_investors_multi`: all four drafts to Indonesian seed funds were judged `unsupported_claim` ("Kejora's focus aligns with your platform's goals")
because the investors' sector focus (climate or fintech) is not in the evidence; the shortlist itself had four borderline sector matches of seven.
The "do not claim alignment" rule is in the prompt but the fact-check did not flag them. Fix: when no evidence item names the sender's sector, the draft must state
only the sender's facts and the ask; add a deterministic check for "aligns with", "fits your focus", "your focus on X" where X is not in the evidence.

### J-09 Shortlists fall short on hard multi-constraint asks (note)
`stress_investors_multi` delivered 7 of 15 and the judge calls 4 of them borderline on sector. Honest, but a user asking for 15 gets 3 good ones. Surface "good: 3, borderline: 4" in the review UI instead of one list.

## Not run (and why)
Eight complex scenarios (journalists with exclusions, Spanish universities, senior hire, neuroscience PhDs, photographers, podcasts, Rust maintainers,
municipal IoT) had not finished when this was written; the first hour was lost to Groq throttling (S-01) and one process runs them sequentially.
`bench` writes its summary JSON only at the end (S-02), so they are not in the repo. Re-run:
`LLM_PROVIDER=openai brain bench --split stress --draft --label stress_full`, then `python tools/llm_judge.py --run stress_full`, and add the findings here.
Also note `tools/llm_judge.py --only` writes `llm_judge_<run>_partial.json`, so two partial judgements of the same run overwrite each other;
the per-scenario numbers above are from the console output.
