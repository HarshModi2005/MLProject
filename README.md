# 🤖 Agentic Autonomous Research Outreach & Cold Email Automation

An end-to-end autonomous agentic system built with LangGraph, Gemini 1.5 Pro, and Streamlit. This agent orchestrates the entire workflow of finding relevant professors for research opportunities, scraping their work, scoring relevance, and generating personalized cold emails.

## Features
- **Conversational Slot-Filling:** Clarifies vague user intent (e.g., "I want to do AI research this summer").
- **Web Crawling:** Pulls data from Serper, Semantic Scholar, and scrapes university faculty pages.
- **Semantic Relevance Scoring:** Embeds user CV and professor publications (using SentenceTransformers & Chroma) to calculate overlap.
- **Personalized Email Generation:** Uses LLM Chain-of-Thought to generate context-aware, non-spammy cold emails referencing specific papers.
- **Supervisor Mode UI:** Streamlit interface to review, edit, and approve drafts before sending.
- **Autonomous Mode Scheduler:** Periodically searches for new professors, handles duplicate detection, and checks for replies via Gmail API.

## Setup

1. **Clone & Virtual Env:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   playwright install chromium
   ```

2. **Environment Variables:**
   Copy `.env.example` to `.env` and fill in API keys:
   ```bash
   cp .env.example .env
   # Edit .env and supply GEMINI_API_KEY, SERPER_API_KEY, TAVILY_API_KEY, and GMAIL OAuth credentials
   ```

3. **Run the Application:**
   Start the interactive CLI for supervised execution:
   ```bash
   python main.py start --mode supervised
   ```

   To review generated emails in the UI:
   ```bash
   python main.py review
   ```

## The brain (new engine)

`brain/` is an outreach agent for any audience: sponsors, customers, investors, employers, journalists,
researchers, candidates, partners, creators, speakers, suppliers. It runs beside the original LangGraph flow
and can hand its drafts to the same review UI.

```bash
python main.py brain run "Find sponsors for our college tech fest in February" --about "..." --upload past_sponsors.csv
python main.py brain run "Find dental clinics in Bangalore for our booking bot" --about "..."
python main.py brain run "Summer research internship on GNNs in Punjab, India" --cv ./cv.pdf
python main.py brain playbooks                 # kinds of outreach and discovery tools
python main.py brain show <id> --drafts        # shortlist, funnel, drafts
python main.py brain resume <id>               # continue after an interruption
python main.py brain suppress someone@x.com    # never contact again
```

### How a campaign runs

1. **Intake and verification** (`brain/intake.py`). The agent restates what it understood, lists what it
   *assumed*, asks only the questions that change the result (each with a default), asks for older lists
   that would help, and shows a verification card — who, where, how many, the facts emails may use (with
   their source), the lists and the assumptions. It loops on corrections until you say yes. ~2 LLM calls.
2. **Uploads** (`brain/uploads.py`). CSV, Excel, JSON, text, PDF or Word. Each is classified (and can be
   corrected) as *targets* to contact, people to *exclude*, good *examples* to find more like, *style*
   (emails you sent before) or *about you* (CV, deck).
3. **What a good prospect is** (`brain/fit.py`). Intake also produces a *prospect profile*: weighted fit
   criteria ("has sponsored college tech fests", "hiring backend engineers", "invests at pre-seed"), timing
   signals, kinds of target to avoid, and the roles to address inside an organisation. Playbooks supply
   defaults; you confirm it on the verification card. Every candidate is scored against it from its
   evidence, without an LLM.
4. **Adaptive discovery** (`brain/explore.py`). Starts with prospects remembered from earlier campaigns
   (free), then the playbook's tools plus any tool that can prove a criterion no listed tool covers.
   Later rounds move the remaining search budget to whichever tool has produced the most *qualified*
   prospects per search (a bandit), and expand from the best finds: lookalike organisations, co-authors.
   It stops when there are enough qualified prospects, not just enough results.
   Tools (`brain/tools.py`, `brain/channels.py`):
   `openalex` (researchers by publication), `coauthors` (OpenAlex co-author graph from the best finds),
   `orgs` (organisations via web search, list pages and event pages), `places` (local businesses via Google
   Maps), `events` (sponsors or speakers of similar past events), `jobs` (companies hiring now, from public
   Greenhouse/Lever/Ashby boards), `portfolio` (investors whose portfolio pages match), `podcasts` (iTunes
   directory; host email from the RSS feed), `news` (journalists via bylines), `github` (developers), `web`
   (people on team pages), `lookalike`, `memory`, plus your own lists.
5. **Search planner and triage** (`brain/searchplan.py`, `brain/triage.py`). Every search hit is triaged
   (the organisation's own site / a list / an event / an article) before anything is fetched. Query yield
   is measured in qualified finds. Templates that worked in earlier campaigns of the same playbook are
   tried early. Queries are never repeated.
6. **Rank → enrich → judge → draft → fact-check**. Hard filters, then a homepage-only *light read* of the
   top organisations before pre-ranking. Enrichment finds contact details and, inside organisations,
   the decision-maker by role. The batched judge rates each confirmed criterion and accounts for how
   reachable the address is. If too few candidates pass, **one backfill round** searches again, seeded
   with the winners and the judge's rejection reasons. Drafts may only use sourced facts and get
   deterministic plus LLM checks.

### Measuring discovery

```bash
python main.py brain bench                          # all scenarios in evals/scenarios (stops after judging)
python main.py brain bench --only sponsors_tech_fest --label after_change --compare evals/results/baseline.json
python main.py brain bench --replay                 # re-run from recorded answers: no search/fetch spend
python main.py brain labels <campaign_id> --out labels.json   # your review decisions as eval labels
```

Each scenario is a ready campaign spec, so intake spends no LLM calls. Each one records its network
answers in `evals/cache/<name>.sqlite`. Metrics: good@K (judge score ≥ 0.7), precision@K, distinct
organisations, share of the shortlist with direct proof (sponsored / hiring / portfolio …), reachability,
search and LLM spend, and USD per good prospect.

### Production hardening

Address trust tiers (`verified` / `attributed` / `inbox` / `guess`), mailbox verification, a Hunter-first route for
named people in companies, a shortlist that never pads, a calibrated score, prompt-injection defences, outcome logging
(`brain outcome`, `brain outcomes`) and per-user workspaces (`brain --workspace`). Held-out scenarios, repeat runs and a
terminal labeller (`brain bench-label`) make the benchmark honest. See [docs/production-hardening.md](docs/production-hardening.md).

### Grounded context

`brain/context.py` holds the agent's working memory: the confirmed understanding, sourced facts about the
sender (your words, CV, uploads — never a model's guess), examples, exclusions, past emails, the search log
and observations made during the run. Every judge and writer prompt is built from it.

### Adapting it

- **New kind of outreach:** add a playbook — a JSON file in `./playbooks/` with the fields of
  `brain/playbooks.py` (description, target kind, tools, roles, inbox preferences, rubric, writing notes,
  which lists to ask for, which questions to ask). No code changes.
- **New data source:** write one function and register it with `@register(...)` in `brain/tools.py`,
  declaring which fit signals it `proves`. The explorer will use it whenever a campaign needs that proof.
- **Knobs:** `BRAIN_TRIAGE_LLM=0` (heuristic triage only), `BRAIN_LIGHT_READ_FACTOR`, `BRAIN_MAX_PEOPLE_SEARCHES`,
  `BRAIN_EXPLORE_ROUNDS`, `BRAIN_BACKFILL=0`, `BRAIN_MEMORY=0` (memory is shared across all campaigns in a
  database; scope it per account before a multi-user deployment).
- **Other LLM provider:** `LLM_PROVIDER=groq|openai`, or any OpenAI-compatible `LLM_BASE_URL`.

Tests: `python -m pytest tests/brain` (offline; no keys needed).

## Folder Structure
- `brain/`: New discovery, ranking and drafting engine (see above).
- `agents/`: LangGraph nodes (Intent, Planning, Crawler, Email).
- `core/`: Relevance scoring, vector store, CV parsing, embeddings.
- `crawler/`: Search APIs and faculty page HTML scraping.
- `db/`: SQLite database models and sessions.
- `email_dispatch/`: Gmail OAuth setup and bulk sending utilities.
- `scheduler/`: Autonomous scheduled tasks for follow-ups and reply-checking.
- `ui/`: Streamlit review dashboard.
- `prompts/`: System instructional prompts for LLMs.

<!-- harmless change -->
