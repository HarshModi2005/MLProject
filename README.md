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
3. **Discovery tools** (`brain/tools.py`), chosen by the playbook: `openalex` (researchers by publication),
   `orgs` (organisations via web search and list pages), `places` (local businesses via Google Maps, with
   website and address), `news` (journalists via article bylines), `github` (developers), `web` (people on
   team pages), plus your own lists.
4. **Search planner** (`brain/searchplan.py`). A Serper budget sized to the campaign; queries run in rounds,
   stop as soon as there are enough candidates, expand only from queries that produced results, and never
   repeat. Every query's yield is logged in the campaign context.
5. **Rank → enrich → judge → draft → fact-check**, as before: hard filters (location, exclusions,
   do-not-contact), contact details from the organisation's own site or the person's pages, a batched LLM
   judge with the playbook's rubric, drafts that may only use sourced facts, and deterministic plus LLM checks.

### Grounded context

`brain/context.py` holds the agent's working memory: the confirmed understanding, sourced facts about the
sender (your words, CV, uploads — never a model's guess), examples, exclusions, past emails, the search log
and observations made during the run. Every judge and writer prompt is built from it.

### Adapting it

- **New kind of outreach:** add a playbook — a JSON file in `./playbooks/` with the fields of
  `brain/playbooks.py` (description, target kind, tools, roles, inbox preferences, rubric, writing notes,
  which lists to ask for, which questions to ask). No code changes.
- **New data source:** write one function and register it with `@register(...)` in `brain/tools.py`.
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
