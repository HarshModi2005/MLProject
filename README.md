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

## Folder Structure
- `agents/`: LangGraph nodes (Intent, Planning, Crawler, Email).
- `core/`: Relevance scoring, vector store, CV parsing, embeddings.
- `crawler/`: Search APIs and faculty page HTML scraping.
- `db/`: SQLite database models and sessions.
- `email_dispatch/`: Gmail OAuth setup and bulk sending utilities.
- `scheduler/`: Autonomous scheduled tasks for follow-ups and reply-checking.
- `ui/`: Streamlit review dashboard.
- `prompts/`: System instructional prompts for LLMs.
