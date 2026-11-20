# Intent understanding and email generation — plan

**Goal:** the campaign the system runs is exactly the one the user meant, and every email it writes is one the
user would send unedited: specific to the recipient, honest about the sender, one clear ask, deliverable.

Status: plan (2026-11-17) · Base: `discovery-v2` @ d9879af · Precedes the final live run.

---

## 1. Where things stand

**Intake** (`brain/intake.py`, ~2 LLM calls): one *understand* call fills the whole spec (playbook, objective, ask,
persona, topics, locations, criteria, anti-patterns, queries, sender facts, assumptions, ≤4 questions with
defaults); deterministic *validate* adds questions for missing topics / ask / sender facts; *refine* folds in
answers, uploads and edits; a verification card loops until the user says yes.

**Drafting** (`brain/draft.py`, 1 strong call per email + batched fact-check): the model writes only the body
(salutation and sign-off are code); deterministic checks (length, filler, placeholders, quoted titles and
numbers must exist in the facts, employer/project conflation); one retry with the failures; a batched LLM
fact-check rewrites flagged drafts once; drafts that still fail are kept and marked for review.

That design is sound — grounded facts, code-owned salutations, checks that cost nothing. The gaps:

### Intent gaps (found in the code)

| # | Gap | Effect |
|---|---|---|
| I1 | The intake prompt and `_SIGNALS` list only the 11 original signals | Intake can never ask for the proof the new sources give (app_publisher, grant_awarded, open_tender, maintains_package, recently_funded, uses_tech, exhibits_at, warm_path...), so the router never adds google_play, nsf_awards, etc. from a real request; only hand-written scenarios did |
| I2 | 28 playbooks are offered as a flat list, one pick, no confidence | Near-neighbours (b2b_sales / distributors / partnerships / integrations; job_search / campus_recruiting) get mis-picked silently |
| I3 | The "ask" is one free-text sentence | The writer cannot tell a 15-minute call from "apply via this link" from "forward to the right person"; no link, availability, deadline or alternative ask for shared inboxes |
| I4 | No model of the *offer* separate from the sender | B2B, sponsorship and fundraising emails need value proposition, proof points (customers, numbers), price/terms; today these hide in free-text facts |
| I5 | No structured constraints | Company size, funding stage, seniority, "no agencies", "remote ok", radius around a city — all lost or left to topic words |
| I6 | `sender_from_text` keeps only 7 highlights although the prompt asks for 6–10 | Relevant experience silently dropped for senders with varied CVs |
| I7 | Validation checks presence only | Topics that describe the sender instead of the targets, an ask without a verb, criteria no source can prove, contradictions (50 targets in one small town) go through |
| I8 | Location strings are used as typed | "Bengaluru" vs "Bangalore", "NCR", "Bay Area", "remote" handled by luck (region_terms) |
| I9 | Nothing persists about the user between campaigns | Signature, tone, standing exclusions, typical asks are re-asked or lost |
| I10 | The bench feeds ready-made specs | No measurement of intake at all; an intake regression would be invisible |

### Email gaps

| # | Gap | Effect |
|---|---|---|
| E1 | "Why you" is the judge's hook only | The email rarely says *why now* (hiring, just raised, recent article) or which confirmed criterion they meet, though both are already computed (timing signals, `Contact.criteria`) |
| E2 | Sender facts are chosen by BM25 overlap | The judge already reads sender and recipient together but its choice of the relevant sender fact is thrown away |
| E3 | One prompt for every situation | A shared inbox, a named decision-maker reached through a shared inbox, a warm path, a guessed address and a person's own address need different openings and different honesty rules |
| E4 | Warm paths are evidence, with no rule | The writer may overclaim ("Priya suggested I write") instead of the true "we're both connected to Priya" |
| E5 | One subject, barely checked | Subjects are often generic; no spam-word, length or specificity check |
| E6 | No deliverability or compliance pass | Spam triggers, link counts, ALL CAPS, missing opt-out line on commercial mail (CAN-SPAM / GDPR legitimate-interest expectations) |
| E7 | No "is this personalised?" check | A fluent but generic email passes every check |
| E8 | English only | Recipients in non-English markets get English mail by default |
| E9 | No follow-ups | Most replies come from the second or third touch |
| E10 | No quality measurement | Checks find errors; nothing scores how *good* an email is, and edits users make teach nothing |

---

## 2. Plan

Ordered by value per LLM dollar. Every step keeps the current guarantees (grounded facts, code-owned
salutation, deterministic checks first).

### Step 1 — Intent: connect intake to everything discovery can now do  *(I1, I2, I6)*
- Build the signal list in the intake schema from `brain/discovery/signals.py`, with a one-line "proved by"
  per signal (from the source manifests), so criteria can name any registered proof.
- Playbook choice returns `playbook`, `confidence` (0–1) and `alternatives` (top 2 with one-line reasons).
  Confidence < 0.6 → one question "Is this closer to A or B?" with the top pick as default; the card shows
  the alternatives.
- Group the catalog by family (research / companies / people / funding / media / jobs / local) so the model
  compares near-neighbours side by side.
- Keep up to 10 sender highlights (fix the `[:7]` cut).
- Files: `intake.py` (schema, `_SIGNALS` → registry), `playbooks.py` (`family` field), `spec.py`.
- Cost: none extra (same calls, longer prompt by ~300 tokens).

### Step 2 — Intent: a structured ask, offer and constraints  *(I3, I4, I5, I8)*
- `CampaignSpec.ask` becomes `Ask {type: call|meeting|reply|apply|intro|review|other, text, link,
  availability, deadline, inbox_text}`; `inbox_text` is the version for shared inboxes ("could you point
  me to whoever handles sponsorships?").
- New `Offer {value_prop, proof_points[], terms, audience}` filled only from user / upload statements (each
  proof point recorded as a sourced fact like today's sender facts).
- New `Constraints {size: [min,max], stages[], seniority, remote_ok, radius_km, must[], avoid[]}`; `avoid`
  merges into anti-patterns, `size`/`stage` become filters where sources provide them (job boards, registries,
  Places reviews as a size proxy) and judge criteria otherwise.
- Location normaliser (`textutil.normalise_location`): aliases (Bengaluru/Bangalore, Gurugram/Gurgaon, NCR,
  Bay Area…), "remote", country inference; the card shows the normalised form.
- Backward compatible: old specs load with `Ask.text = ask`.
- Files: `models.py`, `intake.py`, `textutil.py`, `judge.py` (brief), `draft.py` (prompt).

### Step 3 — Intent: smarter validation and memory of the user  *(I7, I9)*
- Deterministic validators (no LLM): topics overlapping the sender's own skills/product more than the
  persona → "these describe you, not your targets?"; ask without an action verb; criteria whose signal no
  usable source proves (warn, keep as judge-only); target count vs. geography size (Places count as a
  free sanity check is optional); exclusions that match the seeds.
- `UserProfile` in the store: name, signature block, default tone, standing exclusions, typical asks,
  sender facts confirmed before. Pre-fills intake; the card shows what came from memory; `brain profile`
  shows/edits it.
- Files: `intake.py` (validate), `store.py`, `cli.py`.

### Step 4 — Email: use what the pipeline already knows  *(E1, E2, E4)*
- The judge returns `sender_ref` (index of the most relevant sender fact) and `why_now` (a timing signal from
  evidence, or null) in the same call; the writer gets: hook, why-now, the confirmed criterion with the
  highest support, and the judge-chosen sender fact first. **Zero extra calls.**
- Warm-path rule in the prompt and a deterministic check: a connection may be mentioned only as it is in the
  evidence ("we are both connected to X" / "I know X at your company"); claims of an introduction or
  recommendation are rejected unless the user provided one.
- Files: `judge.py`, `draft.py`.

### Step 5 — Email: situation-aware writing  *(E3, E5)*
- A `situation` computed in code: `person_own_address`, `person_guessed_address`, `decision_maker_via_inbox`,
  `org_inbox`, `warm_direct`, `warm_intro`. Each adds a short block of rules to the prompt (e.g. via inbox:
  "address the named person, say in one line why you are writing to the team address, offer the inbox_text
  ask"; guessed address: "no claims that you know them; ask to be pointed to the right person if not").
- Subjects: the writer returns 3; code picks the first that passes: ≤ 8 words, contains a recipient-specific
  term (from hook/evidence) or the ask, no spam words, no ALL CAPS / "!" / "Re:" / "Fwd:".
- Files: `draft.py`.

### Step 6 — Email: quality, deliverability and compliance checks  *(E6, E7)*
- Deterministic, added to `check()`:
  - personalisation: at least one content term from the recipient's evidence or hook appears in the body;
  - one ask: exactly one question mark or call-to-action sentence in the last paragraph;
  - deliverability: spam-trigger list, ≤ 1 link, no URL shorteners, no ALL-CAPS words, ≤ 1 "!";
  - readability: average sentence length ≤ 25 words, no paragraph over 4 sentences.
- Compliance footer for commercial playbooks (b2b_sales, distributors, integrations, brand_deals,
  customer_research, suppliers): a plain opt-out line ("If this isn't relevant, reply 'no' and I won't
  write again.") added in code, with the sender's postal address if the user provided one; a suppression
  reply handler already exists (`brain_suppression`).
- Files: `draft.py`, `playbooks.py` (`commercial: bool`).

### Step 7 — Email: language and follow-ups  *(E8, E9)*
- `Ask.language` (default from the recipient's country, overridable): the writer drafts in that language; the
  fact-check runs on the same text; deterministic checks that are English-specific are skipped for other
  languages.
- Follow-ups: `brain followups <campaign>` drafts up to two short follow-ups per sent email, each with a new
  angle from unused evidence (a second paper, the timing signal), never repeating the first email, same
  checks; scheduling stays manual (or the existing `scheduler/`).
- Files: `draft.py`, `cli.py`, `pipeline.py` (optional stage).

### Step 8 — Measurement before the final run  *(I10, E10)*
Two cheap, repeatable evaluations, so the changes above are measured, not assumed:

- **Intake eval** (`evals/intake/*.json`, `brain eval-intake`): 30 requests written the way users write them
  (vague, typo'd, multi-part, non-English names, "near me", competitor exclusions, the 10 bench use cases and
  10 of the new playbooks), each with expected `playbook` (+ acceptable alternatives), `target_kind`,
  locations, must-have topic words, ask type, and facts that must NOT appear (invented sender facts).
  Scored deterministically. Cost ≈ 30 understand calls ≈ $0.02, cached afterwards.
- **Draft eval** (`brain bench-drafts --run v3_live`): drafts for the top 3 shortlisted contacts of each
  bench scenario, from the stored checkpoints — **no discovery spend**. Scored by (a) the deterministic
  checks, (b) the fact-check, (c) one batched rubric grader on the fast model (specificity, relevance of
  the sender fact, clarity of the ask, tone, would-send yes/no), and (d) a side-by-side with the previous
  drafter on the same contacts. Cost ≈ 30 drafts + 8 checks + 4 grader calls ≈ $0.05.
- **Edits teach the writer:** when the user edits a draft before sending (review flow), the diff is stored;
  the latest 3 approved-after-edit emails become style examples for that user (today only uploaded emails are).

### Step 9 — The final run
- Bench scenarios get an optional `"prompt"`: the bench then runs the intake non-interactively from that
  prompt (1–2 calls each) instead of the ready spec, so the final run tests **request → spec → discovery →
  shortlist → drafts** end to end.
- One live `brain bench --draft` on all 10 scenarios. Expected cost: discovery ≈ $0.06 + intake ≈ $0.02 +
  drafts and checks ≈ $0.10 → **≈ $0.18 LLM** and ≈ 150 searches.
- Report: good@K (judge), label precision where labels exist, reachable, intake accuracy, draft pass rate,
  rubric scores, would-send rate, cost per sendable email.

---

## 3. Order and budget

| Step | LLM spend to build/test | Depends on |
|---|---|---|
| 1 Intake ↔ sources, playbook confidence | 0 (tests are scripted) | — |
| 2 Ask / offer / constraints / locations | 0 | 1 |
| 3 Validation + user profile | 0 | 2 |
| 4 Judge → writer hand-off, warm rule | 0 | — |
| 5 Situations, subject choice | 0 | 4 |
| 6 Quality / deliverability / compliance checks | 0 | 5 |
| 7 Language, follow-ups | 0 | 5 |
| 8 Intake eval + draft eval | ≈ $0.07 (once) | 1–7 |
| 9 Final live run with intake and drafts | ≈ $0.18 | 8 |

Steps 1–7 are built and tested with scripted LLMs (like today's tests); real spend happens only in steps 8
and 9, once each.

## 4. Risks

| Risk | Mitigation |
|---|---|
| A bigger intake schema makes the model fill fields with guesses | Every new field has "null unless stated"; guesses go to `assumptions` and show on the card; the intake eval checks for invented facts |
| More prompt rules make drafts stiff | Situations add only the 2–4 rules that apply; the rubric grader and side-by-side catch regressions |
| Deterministic checks reject good emails (false positives) | Each check is measured on the draft eval; any check that fails > 20% of drafts the grader rates "would send" is softened |
| Compliance wording varies by country | One neutral opt-out line by default; the user can override it in their profile |
| Rubric grader is an LLM grading an LLM | Combined with deterministic checks, side-by-side, and your labels/edits; the grader's verdicts are reported, not used to auto-send |
