# 🤖 Agentic Autonomous Research Outreach & Cold Email Automation
## Implementation Plan

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Module Breakdown](#4-module-breakdown)
5. [Data Models & Schema](#5-data-models--schema)
6. [Phase-wise Development Plan](#6-phase-wise-development-plan)
7. [File & Folder Structure](#7-file--folder-structure)
8. [API & Integration Design](#8-api--integration-design)
9. [Agent Orchestration Design](#9-agent-orchestration-design)
10. [Risk & Mitigation](#10-risk--mitigation)
11. [Future Extensions](#11-future-extensions)

---

## 1. Project Overview

### Goal
Build an end-to-end autonomous agentic system that:
- Accepts a vague high-level user goal (e.g., *"I want to cold mail professors for a research internship"*)
- Clarifies intent through a conversational slot-filling loop
- Plans and executes a structured web research + email outreach workflow
- Operates in supervised mode first, then can transition to fully autonomous mode

### Key Capabilities
| Capability | Description |
|---|---|
| Intent Extraction | Parse vague NL goals into structured state |
| Slot-filling Dialogue | Conversational clarification of missing fields |
| Web Crawling | Scrape faculty pages, labs, and academic directories |
| Relevance Scoring | Rank professors using embedding + keyword similarity |
| Email Generation | Personalized, research-aligned cold emails via LLM |
| Supervised Execution | Human-in-the-loop for first-pass review |
| Autonomous Mode | Periodic crawl, duplicate dedup, reply tracking, follow-ups |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                             │
│              (CLI / Streamlit / Gradio / Web App)                       │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR AGENT (LangGraph / AutoGen)            │
│   Manages agent state, transitions, tool calls, and human-in-the-loop  │
└───┬──────────────┬──────────────┬──────────────┬──────────────┬────────┘
    │              │              │              │              │
    ▼              ▼              ▼              ▼              ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
│ Intent │  │ Planning │  │ Crawler  │  │  Email   │  │  Autonomous  │
│ Agent  │  │  Agent   │  │  Agent   │  │  Agent   │  │  Scheduler   │
└────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────────┘
    │              │              │              │              │
    ▼              ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOOL LAYER                                      │
│  Web Search | Scraper | Embeddings | SMTP/Gmail API | DB | Calendar     │
└─────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       PERSISTENCE LAYER                                 │
│         SQLite / PostgreSQL  +  Vector DB (ChromaDB / Pinecone)         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Agent Framework** | LangGraph (preferred) or AutoGen | Stateful graph-based agent orchestration, human-in-the-loop support |
| **LLM Backend** | OpenAI GPT-4o / Gemini 1.5 Pro / Ollama (local) | Flexible; swap based on cost & privacy needs |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) | Fast, local, no API cost |
| **Web Scraping** | `crawl4ai` + `BeautifulSoup4` + `Playwright` | JavaScript-rendered pages + static HTML |
| **Search API** | Serper API / Google Custom Search / Tavily | Finding professor pages |
| **Vector DB** | ChromaDB (local) or Pinecone (cloud) | Semantic similarity search for relevance scoring |
| **Relational DB** | SQLite (dev) → PostgreSQL (prod) | Professor records, email tracking |
| **Email Sending** | Gmail API (OAuth2) / SMTP / SendGrid | Authenticated email dispatch |
| **Scheduling** | APScheduler / Celery + Redis | Autonomous periodic crawl & follow-up |
| **UI** | Streamlit (MVP) → React (prod) | Quick prototyping, then full UI |
| **CV Parsing** | `PyMuPDF` + LLM summarization | Extract user profile from PDF CV |
| **Config & Secrets** | `python-dotenv` + `.env` file | Secure credential management |
| **Testing** | `pytest` + `unittest.mock` | Unit & integration testing |

---

## 4. Module Breakdown

### Module 1: Intent & Clarification Agent
**File:** `agents/intent_agent.py`

**Responsibilities:**
- Accept raw NL input
- Extract entities (domain, country, timeline, etc.)
- Track slot-filling state as a Pydantic model
- Generate targeted questions for missing slots
- Confirm final structured objective

**Internal State (Pydantic Model):**
```python
class UserIntent(BaseModel):
    goal: str                          # "research_internship"
    research_domains: list[str] = []   # ["machine learning", "NLP"]
    countries: list[str] = []          # ["USA", "Germany"]
    target_institutions: list[str] = []
    timeline: str | None = None        # "Summer 2025"
    user_name: str | None = None
    user_profile_summary: str | None = None  # extracted from CV
    cv_path: str | None = None
    email_tone: str = "professional"
    follow_up_days: int = 7
    outreach_per_week: int = 10
    confirmed: bool = False
```

**Key Tools:**
- `extract_intent(text)` → LLM-based NER + classification
- `identify_missing_slots(state)` → returns list of missing required fields
- `generate_question(slot)` → crafts a natural follow-up question
- `parse_cv(pdf_path)` → extracts user background from CV PDF

---

### Module 2: Planning Agent
**File:** `agents/planning_agent.py`

**Responsibilities:**
- Convert confirmed `UserIntent` → `ExecutionPlan`
- Determine search keywords, filters, scoring criteria
- Select email strategy
- Present plan to user for approval

**Output Schema:**
```python
class ExecutionPlan(BaseModel):
    search_keywords: list[str]
    country_filters: list[str]
    institution_filters: list[str]
    relevance_criteria: dict         # recency, lab_active, keyword_weight
    email_strategy: dict             # tone, personalization_level, ref_pubs
    sending_schedule: dict           # emails_per_week, follow_up_days
    top_k_professors: int = 20
```

---

### Module 3: Web Crawler & Data Extraction Agent
**File:** `agents/crawler_agent.py`

**Responsibilities:**
- Search academic directories / Google Scholar
- Crawl faculty pages and lab sites
- Extract structured professor profiles
- Handle JavaScript-rendered pages via Playwright
- Rate-limit and respect robots.txt

**Professor Profile Schema:**
```python
class ProfessorProfile(BaseModel):
    id: str                     # UUID
    name: str
    email: str | None
    institution: str
    department: str
    lab_url: str | None
    research_keywords: list[str]
    recent_publications: list[dict]  # [{title, year, url}]
    accepting_students: bool | None  # inferred from page text
    crawled_at: datetime
    raw_page_text: str
```

**Key Crawl Targets:**
1. Google Scholar search
2. University CS/AI department pages
3. Individual faculty profile pages
4. DBLP / Semantic Scholar / arXiv author pages
5. Lab group pages

---

### Module 4: Relevance Scoring Engine
**File:** `core/relevance_scorer.py`

**Scoring Formula:**
```
relevance_score = (
    0.35 * keyword_overlap_score +
    0.35 * embedding_similarity_score +
    0.15 * recency_score +           # based on latest pub year
    0.10 * geographic_score +        # matches user country preference
    0.05 * student_acceptance_score  # if page indicates open positions
)
```

**Implementation Details:**
- Embed user's research interest summary using `sentence-transformers`
- Embed professor's recent publications + research keywords
- Compute cosine similarity
- Normalize all sub-scores to [0, 1]
- Store embeddings in ChromaDB for fast retrieval

---

### Module 5: Email Generation Agent
**File:** `agents/email_agent.py`

**Responsibilities:**
- For each shortlisted professor, generate a personalized cold email
- Pull professor's latest paper titles + research themes
- Reference specific professor work in the email body
- Inject user profile / CV summary
- Output email as structured object

**Email Schema:**
```python
class DraftEmail(BaseModel):
    professor_id: str
    recipient_name: str
    recipient_email: str
    subject: str
    body: str                  # HTML or plaintext
    personalization_notes: str # why this email is personalized
    draft_status: str          # "draft" | "approved" | "sent"
    created_at: datetime
```

**Prompt Strategy (Chain-of-Thought):**
1. "Here are the professor's recent papers: {papers}"
2. "Here is the student's background: {user_profile}"
3. "Write a 200–250 word email that references specific work..."

---

### Module 6: Supervised Execution UI
**File:** `ui/review_ui.py` (Streamlit)

**Features:**
- Show list of shortlisted professors + relevance scores
- Display extracted research summaries
- Show generated draft email per professor
- Allow user to: Approve / Edit / Reject each email
- Show email preview (rendered HTML)
- Bulk approve / reject option
- Export draft emails as CSV

---

### Module 7: Autonomous Execution Scheduler
**File:** `scheduler/autonomous_scheduler.py`

**Responsibilities:**
- Periodic re-crawl for new professors matching the plan
- Duplicate detection before any new outreach
- Email dispatch via Gmail API
- Reply monitoring (Gmail API / IMAP)
- Follow-up scheduler (send follow-up if no reply in N days)
- Logging all events to DB

**APScheduler Jobs:**
```python
scheduler.add_job(crawl_new_professors,   'interval', weeks=1)
scheduler.add_job(check_replies,          'interval', hours=6)
scheduler.add_job(send_followups,         'interval', days=1)
```

---

### Module 8: Database Layer
**File:** `db/models.py` + `db/database.py`

**Tables:**
```sql
-- Professors table
CREATE TABLE professors (
    id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    institution TEXT,
    department TEXT,
    lab_url TEXT,
    research_keywords TEXT,   -- JSON array
    recent_publications TEXT, -- JSON array
    accepting_students BOOLEAN,
    relevance_score REAL,
    crawled_at TIMESTAMP
);

-- Emails table
CREATE TABLE emails (
    id TEXT PRIMARY KEY,
    professor_id TEXT REFERENCES professors(id),
    subject TEXT,
    body TEXT,
    status TEXT,  -- draft | approved | sent | replied | followed_up
    sent_at TIMESTAMP,
    reply_received_at TIMESTAMP,
    follow_up_count INTEGER DEFAULT 0,
    created_at TIMESTAMP
);

-- User sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    intent_json TEXT,         -- serialized UserIntent
    plan_json TEXT,           -- serialized ExecutionPlan
    mode TEXT,                -- "supervised" | "autonomous"
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

## 5. Data Models & Schema

*(See individual module schemas above)*

**Core State Object (passed across agents):**
```python
class AgentState(TypedDict):
    messages: list[BaseMessage]      # conversation history
    intent: UserIntent | None
    plan: ExecutionPlan | None
    professors: list[ProfessorProfile]
    draft_emails: list[DraftEmail]
    mode: str                         # "clarification" | "planning" | "crawl" 
                                      # | "review" | "autonomous"
    human_feedback: str | None
```

---

## 6. Phase-wise Development Plan

### Phase 1 – Foundation (Week 1–2)
- [ ] Set up project folder structure
- [ ] Configure environment (`.env`, requirements, virtual env)
- [ ] Implement `UserIntent` and `AgentState` Pydantic models
- [ ] Build basic slot-filling conversational loop (CLI)
- [ ] Implement CV PDF parsing with PyMuPDF + LLM summarization
- [ ] Test intent extraction with 5–10 example inputs

**Deliverable:** Working CLI that takes vague goal → fills all slots → prints confirmed `UserIntent`

---

### Phase 2 – Planning & Crawling (Week 3–4)
- [ ] Implement `ExecutionPlan` generation with LLM
- [ ] Build web crawler using `crawl4ai` + `BeautifulSoup`
- [ ] Add Playwright for JS-rendered pages
- [ ] Integrate Serper/Tavily API for initial professor discovery
- [ ] Implement Semantic Scholar / DBLP scraping
- [ ] Build `ProfessorProfile` extraction pipeline
- [ ] Store profiles in SQLite

**Deliverable:** Given a plan, the crawler returns N professor profiles stored in DB

---

### Phase 3 – Relevance Scoring (Week 5)
- [ ] Integrate `sentence-transformers` for embeddings
- [ ] Set up ChromaDB for vector storage
- [ ] Implement multi-factor scoring formula
- [ ] Build Top-K selection logic
- [ ] Test with sample professor datasets

**Deliverable:** Given a list of profiles + user intent → returns ranked Top-K professors

---

### Phase 4 – Email Generation (Week 6)
- [ ] Design prompt templates for cold email
- [ ] Implement chain-of-thought email generation
- [ ] Add personalization engine (references specific papers)
- [ ] Build subject line generator
- [ ] Output structured `DraftEmail` objects
- [ ] Test across 10 diverse professor profiles

**Deliverable:** Given Top-K professors + user profile → generates N personalized draft emails

---

### Phase 5 – Supervised UI (Week 7)
- [ ] Build Streamlit review dashboard
- [ ] Professor list with relevance scores
- [ ] Email preview + edit + approve/reject UI
- [ ] CV attachment support
- [ ] Export approved emails as CSV

**Deliverable:** User can review, edit, and approve emails through a visual interface

---

### Phase 6 – Email Dispatch (Week 8)
- [ ] Integrate Gmail API (OAuth2 authentication)
- [ ] Implement email sending with attachment support
- [ ] Implement rate limiting (N emails/day)
- [ ] Update DB status on send
- [ ] Test with real email addresses

**Deliverable:** Approved emails are dispatched via Gmail API with proper logging

---

### Phase 7 – Autonomous Mode (Week 9–10)
- [ ] Implement APScheduler for periodic tasks
- [ ] Build reply monitoring via Gmail API (IMAP/label check)
- [ ] Implement follow-up email logic
- [ ] Build duplicate detection (by professor email + previous outreach)
- [ ] Add re-crawl scheduler
- [ ] Implement mode-switching (supervised → autonomous)

**Deliverable:** System runs autonomously, sends emails, checks replies, sends follow-ups

---

### Phase 8 – LangGraph Orchestration (Week 11–12)
- [ ] Refactor all agents into LangGraph nodes
- [ ] Define graph edges and conditional transitions
- [ ] Add human-in-the-loop interrupt nodes
- [ ] End-to-end integration tests
- [ ] Error handling & retry logic across all agents

**Deliverable:** Full LangGraph-orchestrated system running end-to-end

---

### Phase 9 – Polish & Productionize (Week 13–14)
- [ ] Migrate from SQLite to PostgreSQL
- [ ] Dockerize the application
- [ ] Add logging (structured JSON logs)
- [ ] Add monitoring / alerting (optional: Grafana)
- [ ] Write documentation + README
- [ ] Record demo video

**Deliverable:** Production-ready system with Docker Compose setup

---

## 7. File & Folder Structure

```
autonomous-research-outreach/
│
├── .env                          # API keys and secrets (never commit)
├── .env.example                  # Template for env variables
├── requirements.txt
├── README.md
├── IMPLEMENTATION_PLAN.md        # ← This document
│
├── main.py                       # Entry point (CLI or Streamlit launcher)
│
├── agents/
│   ├── __init__.py
│   ├── intent_agent.py           # Slot-filling conversation agent
│   ├── planning_agent.py         # ExecutionPlan generation
│   ├── crawler_agent.py          # Web crawling and extraction
│   └── email_agent.py            # Personalized email generation
│
├── core/
│   ├── __init__.py
│   ├── relevance_scorer.py       # Multi-factor scoring engine
│   ├── cv_parser.py              # PDF CV parsing + LLM summarization
│   ├── embeddings.py             # sentence-transformers wrapper
│   └── vector_store.py           # ChromaDB interface
│
├── crawler/
│   ├── __init__.py
│   ├── search_api.py             # Serper / Tavily integration
│   ├── faculty_scraper.py        # BeautifulSoup + crawl4ai scraping
│   ├── scholar_scraper.py        # Google Scholar / Semantic Scholar
│   └── profile_extractor.py     # LLM-based structured extraction
│
├── db/
│   ├── __init__.py
│   ├── database.py               # SQLAlchemy setup + session factory
│   ├── models.py                 # ORM models (Professor, Email, Session)
│   └── migrations/               # Alembic migration scripts
│
├── email_dispatch/
│   ├── __init__.py
│   ├── gmail_client.py           # Gmail API OAuth2 wrapper
│   ├── smtp_client.py            # Fallback SMTP client
│   └── rate_limiter.py           # Email sending rate control
│
├── scheduler/
│   ├── __init__.py
│   ├── autonomous_scheduler.py   # APScheduler job definitions
│   ├── reply_monitor.py          # Gmail IMAP reply checking
│   └── followup_engine.py        # Follow-up logic + templates
│
├── orchestrator/
│   ├── __init__.py
│   ├── graph.py                  # LangGraph graph definition
│   ├── state.py                  # AgentState TypedDict
│   └── tools.py                  # Shared LangChain tools
│
├── ui/
│   ├── __init__.py
│   ├── review_ui.py              # Streamlit supervised review dashboard
│   └── components/               # Reusable UI components
│
├── prompts/
│   ├── intent_extraction.txt
│   ├── plan_generation.txt
│   ├── email_generation.txt
│   └── followup_email.txt
│
├── tests/
│   ├── test_intent_agent.py
│   ├── test_crawler.py
│   ├── test_relevance_scorer.py
│   ├── test_email_agent.py
│   └── fixtures/                 # Sample data for testing
│
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 8. API & Integration Design

### External APIs Required

| API | Purpose | Free Tier? |
|---|---|---|
| OpenAI / Gemini | LLM for intent, planning, email gen | Paid (GPT-4o) |
| Serper API | Google Search programmatic access | 2,500 free/mo |
| Tavily API | Research-focused web search | 1,000 free/mo |
| Semantic Scholar | Paper metadata + author info | Free |
| Gmail API | Email send + reply monitoring | Free (OAuth) |
| Sentence Transformers | Local embeddings | Free (local) |
| ChromaDB | Local vector database | Free (local) |

### Environment Variables (`.env`)
```bash
# LLM
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...

# Search
SERPER_API_KEY=...
TAVILY_API_KEY=...

# Email
GMAIL_CLIENT_ID=...
GMAIL_CLIENT_SECRET=...
GMAIL_REFRESH_TOKEN=...

# DB
DATABASE_URL=sqlite:///./outreach.db

# Config
MAX_EMAILS_PER_WEEK=10
FOLLOWUP_DAYS=7
TOP_K_PROFESSORS=20
```

---

## 9. Agent Orchestration Design

### LangGraph State Machine

```
[START]
   │
   ▼
[intent_node]  ←──────────────────────────────────┐
   │                                               │
   │  (slots incomplete)                           │
   ▼                                               │
[clarification_node]  ──── human input ────────────┘
   │
   │  (all slots filled)
   ▼
[planning_node]
   │
   ▼
[plan_review_node]  ──── human approval ────┐
   │                                        │ (user requests changes)
   │  (approved)                            └──► [planning_node]
   ▼
[crawl_node]
   │
   ▼
[scoring_node]
   │
   ▼
[email_generation_node]
   │
   ▼
[supervised_review_node]  ──── human review ────┐
   │                                             │ (edit/reject)
   │  (approved)                                 └──► [email_generation_node]
   ▼
[send_email_node]
   │
   ▼
[track_and_monitor_node]  ←─── scheduler triggers
   │
   ▼
[followup_node]
   │
   ▼
[autonomous_mode_node]  (repeats crawl → score → email → send loop)
```

### Human-in-the-Loop Interrupts
LangGraph `interrupt()` is used at:
1. After clarification → before planning (confirm intent)
2. After planning → before crawl (confirm plan)
3. After email generation → before sending (review emails)
4. Before switching to autonomous mode (explicit approval)

---

## 10. Risk & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Email scraping fails (no public email) | High | Use contact forms + institutional email guessing (first.last@uni.edu patterns) |
| Anti-bot / CAPTCHA on university pages | High | Use Playwright with delays, rotate user agents, respect robots.txt |
| Gmail API rate limiting | Medium | Implement exponential backoff + daily quota management |
| LLM hallucinations in email | High | Always ground email content in scraped page data; show to user before send |
| Duplicate outreach to same professor | Medium | DB-level deduplication by email address before every send |
| User CV contains sensitive info | Medium | Process CV locally, never send raw PDF to external LLMs (summarize first) |
| Spam flagging by email providers | High | Warm up email, limit volume, use professional subject lines, avoid spam words |
| Professors mark as spam | High | Ensure high personalization, include unsubscribe notice, limit follow-ups |

---

## 11. Future Extensions

| Feature | Description |
|---|---|
| **Multi-user support** | Separate sessions and credentials per user, SaaS model |
| **LinkedIn Outreach** | Extend crawler to LinkedIn for industry/research connections |
| **Resume Tailoring** | Per-professor CV customization suggestions |
| **Reply Analysis** | Classify replies (interested / rejected / auto-reply) with LLM |
| **Analytics Dashboard** | Track response rate, open rate, best-performing emails |
| **Multi-language emails** | Support German, French, Japanese outreach |
| **Integration with Notion/Sheets** | Export tracking data to Notion DB or Google Sheets |
| **Voice Input** | Accept vocal intent description via Whisper API |

---

## ✅ Next Immediate Steps

1. **Review and confirm** this implementation plan
2. **Set up the project** → create folder structure + virtual environment
3. **Start Phase 1** → build the slot-filling intent agent (CLI first)
4. **Get API keys** → OpenAI (or Gemini), Serper, Gmail OAuth
5. **Test with your own use case** as the first real run

---

*Generated: 2026-03-04 | This is a living document — update as the project evolves.*
