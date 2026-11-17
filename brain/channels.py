"""
Signal-driven discovery channels.

Keyword search finds organisations that *describe themselves* with the right words. The strongest
prospects are usually proven by what they have DONE, and each channel here goes straight to one kind
of proof:

    events     sponsors (or speakers) of similar past events      -> past_sponsorship / spoke_before
    jobs       companies with open roles on public job boards      -> hiring
    portfolio  investors whose portfolio pages match the topic     -> portfolio_fit
    podcasts   (manifest: brain/discovery/sources/podcasts.json)   -> recent_coverage, audience
    lookalike  organisations similar to known-good ones            -> (similarity to winners)
    coauthors  researchers who publish with known-good ones        -> publications
    memory     prospects already found in earlier campaigns        -> (free, instant)

Every channel tags the proof it found in Contact.signals (see brain/fit.py SIGNAL_KEYS) and as sourced
Evidence, so the fit model and the judge can use it and the email can cite it. No LLM calls, except the
speaker fallback in `events`, which reuses the existing page extractor within its budget.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin, urlsplit

from bs4 import BeautifulSoup

from brain.models import CampaignSpec, Contact, EmailCandidate, Evidence
from brain.textutil import clean_name, country_from_domain, looks_like_person_name, region_terms, registrable_domain, squash

log = logging.getLogger("brain.channels")

_SPONSOR_HDR = re.compile(r"\b(sponsors?|sponsored by|partners?|supported by|powered by|backed by|associates?|"
                          r"title sponsor|co-?sponsors?|our partners|in association with)\b", re.I)
_SPEAKER_HDR = re.compile(r"\b(speakers?|judges?|mentors?|panel(ists)?|keynotes?|jury)\b", re.I)
_SOCIAL = ("facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com", "youtube.com", "t.me",
           "whatsapp.com", "discord.gg", "discord.com", "github.com", "medium.com", "google.com", "apple.com",
           "goo.gl", "bit.ly", "forms.gle", "wa.me", "devfolio.co", "unstop.com", "mlh.io", "eventbrite.com")
_GENERIC_ALT = re.compile(r"^(logo|image|img|sponsor|partner|banner|icon|photo|picture|line|arrow|bg|\d+)$", re.I)
_JUNK_NAME = re.compile(r"\b(invite|become|apply|register|join|click|learn more|read more|contact|sponsor us|"
                        r"partner with|our sponsors|speaker|download|brochure|ticket|blog|fact check|news|"
                        r"retake|course|exam)s?\b", re.I)
_DOMAINISH = re.compile(r"^(https?://)?(www\.)?[\w-]+(\.[\w-]+)+/?$", re.I)


# ── events ───────────────────────────────────────────────────────────────────

def _label(domain: str) -> str:
    return domain.split(".")[0].replace("-", " ").title()


def orgs_from_event_page(html: str, url: str, title: str = "", place_terms: Optional[List[str]] = None) -> List[Contact]:
    """Organisations named in an event page's sponsors/partners section (logos and links). Footers are kept,
    since sponsor strips often live there. Each one carries direct proof of past sponsorship — full strength
    when the event itself is in the campaign's area (`place_terms`), half strength otherwise."""
    from brain.orgs import _org_contact, _plausible_org_name, is_aggregator

    soup = BeautifulSoup(html, "lxml")
    for el in soup.find_all(["script", "style", "noscript"]):
        el.decompose()
    own = registrable_domain(url)
    event = squash(title or (soup.title.get_text(" ") if soup.title else "") or own, 120)
    containers = []
    headed = set()  # containers found by a visible "Sponsors" heading (text links trusted there)
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "section", "div", "ul"]):
        attr = " ".join([tag.get("id") or ""] + list(tag.get("class") or []))
        heading = tag.get_text(" ", strip=True)[:80] if tag.name.startswith("h") else ""
        if _SPONSOR_HDR.search(attr) or (heading and len(heading) < 60 and _SPONSOR_HDR.search(heading)):
            box = tag if tag.name in ("section", "div", "ul") else (tag.find_parent(["section", "div"]) or tag.parent)
            if box is not None and len(box.get_text(" ", strip=True)) < 15000 and box not in containers:
                containers.append(box)
                if heading:
                    headed.add(id(box))
    sponsor_page = bool(re.search(r"/(sponsors?|partners?)(/|$|\.)", urlsplit(url).path, re.I))
    if not containers and sponsor_page:
        containers = [soup.body or soup]

    found: Dict[str, Contact] = {}
    page_text = soup.get_text(" ").lower()
    local = not place_terms or any(t in page_text for t in place_terms)
    where = " (an event in the target area)" if place_terms and local else ""

    def push(name: str, domain: Optional[str], link: Optional[str]) -> None:
        name = squash(name, 80)
        bad = (not _plausible_org_name(name) or _SPONSOR_HDR.fullmatch(name.lower()) or _GENERIC_ALT.match(name)
               or _JUNK_NAME.search(name) or _DOMAINISH.match(name))
        if domain:
            if bad:
                name = _label(domain)
        elif bad or len(name) < 3 or not name[0].isupper():
            return  # a logo with no link and no clear name is not worth a website lookup
        key = domain or name.lower()
        if key in found:
            return
        c = _org_contact(name, domain, f"https://{domain}/" if domain else None, source="events")
        c.evidence.append(Evidence(kind="page", text=f"Sponsor or partner of {event}{where}", url=url))
        c.signals["past_sponsorship"] = 1.0 if local else 0.5
        found[key] = c

    for box in containers[:8]:
        for a in box.find_all("a", href=True):
            href = urljoin(url, a["href"].strip())
            dom = registrable_domain(href) if href.startswith("http") else ""
            if not dom or dom == own or is_aggregator(dom) or any(s in dom for s in _SOCIAL):
                continue
            img = a.find("img")
            if img is None and not (sponsor_page or id(box) in headed):
                continue  # a text link in a "partner"-classed block is usually navigation, a blog or a news item
            name = (img.get("alt") or img.get("title") or "") if img else ""
            push(re.sub(r"\s*logo\s*$", "", name or a.get_text(" ", strip=True), flags=re.I), dom, href)
        for img in box.find_all("img", alt=True):
            if img.find_parent("a") is None:
                push(re.sub(r"\s*logo\s*$", "", img["alt"], flags=re.I), None, None)
        if len(found) >= 40:
            break
    return list(found.values())[:40]


def speakers_from_event_page(html: str, url: str, title: str = "") -> List[Contact]:
    """People listed as speakers/judges/mentors on an event page (directory heuristics on that section)."""
    from brain.pages import directory_contacts, parse

    soup = BeautifulSoup(html, "lxml")
    boxes = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "section", "div"]):
        attr = " ".join([tag.get("id") or ""] + list(tag.get("class") or []))
        heading = tag.get_text(" ", strip=True)[:60] if tag.name.startswith("h") else ""
        if _SPEAKER_HDR.search(attr) or (heading and _SPEAKER_HDR.search(heading)):
            box = tag if tag.name in ("section", "div") else (tag.find_parent(["section", "div"]) or tag.parent)
            if box is not None and box not in boxes:
                boxes.append(box)
    event = squash(title or (soup.title.get_text(" ") if soup.title else ""), 120)
    people: Dict[str, Contact] = {}
    for box in boxes[:4] or [soup.body or soup]:
        for c in directory_contacts(parse(str(box), url)):
            if c.name.lower() in people:
                continue
            c.sources = ["events"]
            c.evidence.append(Evidence(kind="page", text=f"Speaker, judge or mentor at {event}", url=url))
            c.signals["spoke_before"] = 1.0
            people[c.name.lower()] = c
    return list(people.values())[:30]


def _place_terms(spec: CampaignSpec) -> List[str]:
    from brain.orgs import _country_names

    terms = [t for r in spec.regions for t in region_terms(r)]
    terms += [n for n, c in _country_names().items() if c in spec.countries]
    return terms


def event_queries(spec: CampaignSpec) -> List[str]:
    place = (spec.regions or [""])[0]
    role = "speakers" if spec.target_kind == "person" else "sponsors"
    out = list(spec.event_queries)
    for t in spec.topics[:3]:
        out.append(squash(f"{t} {role} {place}"))
        out.append(squash(f"{t} hackathon {role} {place}"))
    return list(dict.fromkeys(q for q in out if q))


def discover_events(run) -> Tuple[List[Contact], Dict]:
    """Find similar past events and read who sponsored them (org campaigns) or spoke at them (person campaigns)."""
    from brain.discover import llm_extract
    from brain.orgs import is_aggregator
    from brain.triage import heuristic_kind

    spec, search, fetcher, planner = run.spec, run.deps.search, run.deps.fetcher, run.planner
    people_mode = spec.target_kind == "person" or (not spec.target_kind and run.playbook.target_kind == "person")
    found: Dict[str, Contact] = {}
    stats = {"queries": 0, "event_pages": 0, "found": 0, "llm_extracted_pages": 0}
    start = search.calls
    seen: set = set()

    def execute(query: str):
        stats["queries"] += 1
        hits = [h for h in search.search(query, n=10) if h.url not in seen]
        seen.update(h.url for h in hits)
        pages = [h for h in hits if heuristic_kind(h, is_aggregator) in ("event", "own_site", "unsure", "article")
                 and re.search(r"sponsor|partner|speaker|judge|fest|hackathon|summit|conference|meetup",
                               (h.title + " " + h.snippet + " " + h.url), re.I)][:4]
        new = 0
        for h, page in zip(pages, fetcher.get_many([h.url for h in pages])):
            if not page.ok:
                continue
            stats["event_pages"] += 1
            got = speakers_from_event_page(page.html, page.url, h.title) if people_mode else \
                orgs_from_event_page(page.html, page.url, h.title, place_terms=_place_terms(spec))
            if people_mode and not got and run.deps.llm.available and stats["llm_extracted_pages"] < 3:
                try:
                    got = llm_extract(page.html, page.url, run.deps.llm)
                    stats["llm_extracted_pages"] += 1
                    for c in got:
                        c.signals["spoke_before"] = 0.6
                        c.sources = ["events"]
                except Exception as e:
                    log.warning("speaker extraction failed for %s: %s", page.url, e)
            for c in got:
                key = c.org_domain if c.kind == "org" and c.org_domain else c.name.lower()
                if key in found:
                    found[key].evidence += [e for e in c.evidence if e.text not in {x.text for x in found[key].evidence}]
                    found[key].signals["events_count"] = found[key].signals.get("events_count", 1) + 1
                    continue
                found[key] = c
                new += 1
        stats["found"] = len(found)
        return len(hits), new, new

    def left():
        lim = search.remaining - 2
        return min(lim, run.budget - (search.calls - start)) if run.budget is not None else lim

    planner.run(execute, have=lambda: len(found), budget_left=left, kind="events", first_round_max=5,
                queries=[q for q in event_queries(spec) if not run.ctx.tried(q)][:6])
    return list(found.values()), stats


# ── jobs: public applicant-tracking boards ───────────────────────────────────

_ATS_PATTERNS = [
    ("greenhouse", re.compile(r"(?:boards|job-boards)\.greenhouse\.io/(?:embed/job_board\?for=)?([A-Za-z0-9_-]+)")),
    ("lever", re.compile(r"jobs\.lever\.co/([A-Za-z0-9_.-]+)")),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/([A-Za-z0-9_.%-]+)")),
    ("workable", re.compile(r"apply\.workable\.com/(?!j/|api/)([A-Za-z0-9_-]+)")),
    ("recruitee", re.compile(r"//(?!www\.|jobs\.)([A-Za-z0-9-]+)\.recruitee\.com")),
    ("smartrecruiters", re.compile(r"(?:jobs|careers)\.smartrecruiters\.com/([A-Za-z0-9_-]+)")),
]
# Two grouped searches per topic cover all six boards (verified: Google honours OR between site: filters).
_ATS_SITE_GROUPS = [("boards.greenhouse.io", "jobs.lever.co", "jobs.ashbyhq.com"),
                    ("apply.workable.com", "recruitee.com", "jobs.smartrecruiters.com")]


def ats_token(url: str) -> Optional[Tuple[str, str]]:
    for ats, rx in _ATS_PATTERNS:
        m = rx.search(url)
        if m and m.group(1).lower() not in ("embed", "jobs", "api"):
            return ats, m.group(1)
    return None


def fetch_board(fetcher, ats: str, token: str) -> Tuple[str, List[Dict[str, str]]]:
    """(company name, [{"title", "location", "url", "date"}]) from a public job board API."""
    jobs: List[Dict[str, str]] = []
    name = token.replace("-", " ").replace("_", " ").title()
    if ats == "greenhouse":
        info = fetcher.get_json(f"https://boards-api.greenhouse.io/v1/boards/{token}")
        if isinstance(info, dict) and info.get("name"):
            name = str(info["name"])
        data = fetcher.get_json(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs")
        for j in (data or {}).get("jobs", []) if isinstance(data, dict) else []:
            jobs.append({"title": str(j.get("title") or ""), "location": str((j.get("location") or {}).get("name") or ""),
                         "url": str(j.get("absolute_url") or ""), "date": str(j.get("updated_at") or "")[:10]})
    elif ats == "lever":
        data = fetcher.get_json(f"https://api.lever.co/v0/postings/{token}", params={"mode": "json"})
        for j in data if isinstance(data, list) else []:
            cats = j.get("categories") or {}
            ts = j.get("createdAt")
            date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if isinstance(ts, (int, float)) else ""
            jobs.append({"title": str(j.get("text") or ""), "location": str(cats.get("location") or ""),
                         "url": str(j.get("hostedUrl") or ""), "date": date})
    elif ats == "ashby":
        data = fetcher.get_json(f"https://api.ashbyhq.com/posting-api/job-board/{token}")
        for j in (data or {}).get("jobs", []) if isinstance(data, dict) else []:
            jobs.append({"title": str(j.get("title") or ""), "location": str(j.get("location") or ""),
                         "url": str(j.get("jobUrl") or ""), "date": str(j.get("publishedAt") or "")[:10]})
    elif ats == "workable":
        data = fetcher.get_json(f"https://apply.workable.com/api/v1/widget/accounts/{token}")
        if isinstance(data, dict):
            name = str(data.get("name") or name)
            for j in data.get("jobs") or []:
                where = ", ".join(x for x in (j.get("city"), j.get("country")) if x) + (" (remote)" if j.get("telecommuting") else "")
                jobs.append({"title": str(j.get("title") or ""), "location": where, "url": str(j.get("url") or ""),
                             "date": str(j.get("published_on") or "")[:10]})
    elif ats == "recruitee":
        data = fetcher.get_json(f"https://{token}.recruitee.com/api/offers/")
        for j in (data or {}).get("offers", []) if isinstance(data, dict) else []:
            name = str(j.get("company_name") or name)
            jobs.append({"title": str(j.get("title") or ""), "location": str(j.get("location") or ""),
                         "url": str(j.get("careers_url") or ""), "date": str(j.get("published_at") or "")[:10]})
    elif ats == "smartrecruiters":
        data = fetcher.get_json(f"https://api.smartrecruiters.com/v1/companies/{token}/postings", params={"limit": 100})
        for j in (data or {}).get("content", []) if isinstance(data, dict) else []:
            name = str((j.get("company") or {}).get("name") or name)
            loc = j.get("location") or {}
            where = str(loc.get("fullLocation") or loc.get("city") or "") + (" (remote)" if loc.get("remote") else "")
            jobs.append({"title": str(j.get("name") or ""), "location": where,
                         "url": f"https://jobs.smartrecruiters.com/{token}/{j.get('id')}" if j.get("id") else "",
                         "date": str(j.get("releasedDate") or "")[:10]})
    return name, [j for j in jobs if j["title"]]


def _in_place(text: str, spec: CampaignSpec) -> bool:
    if not spec.regions and not spec.countries:
        return True
    low = (text or "").lower()
    if "remote" in low:
        return True
    from brain.orgs import _country_names

    names = _country_names()
    terms = [t for r in spec.regions for t in region_terms(r)] + [n for n, c in names.items() if c in spec.countries]
    return any(t in low for t in terms)


def _guess_site(fetcher, name: str, token: str) -> Optional[str]:
    """Cheap website guess for a board-only company: try token.com / token.io and accept a page that names it."""
    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) > 2]
    for tld in ("com", "io", "ai"):
        url = f"https://{token.lower()}.{tld}/"
        page = fetcher.get(url)
        if page.ok and words and any(w in page.html[:5000].lower() for w in words):
            return registrable_domain(url)
    return None


def discover_jobs(run) -> Tuple[List[Contact], Dict]:
    """Companies hiring now, from public job boards found by search: Greenhouse, Lever, Ashby, Workable,
    Recruitee and SmartRecruiters (each has a public JSON API for a company's open roles)."""
    from brain.orgs import _org_contact

    spec, search, fetcher, planner = run.spec, run.deps.search, run.deps.fetcher, run.planner
    place = (spec.regions or [""])[0]
    role_words = [r for r in (spec.persona.roles or [])][:2]
    queries = []
    for t in spec.topics[:3]:
        for group in _ATS_SITE_GROUPS:
            queries.append(squash("(" + " OR ".join(f"site:{g}" for g in group) + f") {t} {place}"))
    found: Dict[str, Contact] = {}
    stats = {"queries": 0, "boards": 0, "companies": 0, "in_region": 0}
    start = search.calls
    topic_terms = [w for t in spec.topics + spec.expanded_terms for w in t.lower().split() if len(w) > 3]

    def execute(query: str):
        stats["queries"] += 1
        hits = search.search(query, n=10)
        tokens = list(dict.fromkeys(t for t in (ats_token(h.url) for h in hits) if t))
        new = good = 0
        for ats, token in tokens[:5]:
            key = f"{ats}:{token.lower()}"
            if key in found:
                continue
            name, jobs = fetch_board(fetcher, ats, token)
            stats["boards"] += 1
            if not jobs:
                continue
            local = [j for j in jobs if _in_place(j["location"], spec)]
            if (spec.regions or spec.countries) and not local:
                continue  # hiring, but not where the candidate wants to work
            relevant = [j for j in local if any(w in j["title"].lower() for w in topic_terms + [r.lower() for r in role_words])]
            shown = (relevant or local or jobs)[:4]
            # Guess websites for the first dozen companies only; the rest are resolved later if they rank well.
            dom = _guess_site(fetcher, name, token) if stats["boards"] <= 12 else None
            c = _org_contact(name, dom, f"https://{dom}/" if dom else None, source="jobs")
            c.add_key(f"ats:{key}")
            text = "Hiring now: " + "; ".join(f"{j['title']} ({j['location'] or 'n/a'})" for j in shown)
            c.evidence.append(Evidence(kind="page", text=squash(text, 300), url=shown[0]["url"] or None,
                                       detail=f"{len(jobs)} open roles, {len(local)} in the target area"))
            c.signals["hiring"] = 1.0 if local else 0.5
            c.signals["open_roles"] = float(len(jobs))
            if local:
                c.location_text = squash(local[0]["location"], 120)
                stats["in_region"] += 1
            found[key] = c
            new += 1
            good += bool(local and (relevant or not topic_terms))
        stats["companies"] = len(found)
        return len(hits), new, good

    def left():
        lim = search.remaining - 2
        return min(lim, run.budget - (search.calls - start)) if run.budget is not None else lim

    planner.run(execute, have=lambda: len(found), budget_left=left, kind="jobs",
                queries=[q for q in queries if not run.ctx.tried(q)][:6])
    return list(found.values()), stats


# ── portfolio: investors whose portfolio fits ─────────────────────────────────

_PORTFOLIO_LINK = re.compile(r"portfolio|companies|investments|our-founders|startups", re.I)


def portfolio_evidence(fetcher, org: Contact, topic_terms: List[str]) -> bool:
    """Read an investor's portfolio page and record how much of it matches the topics."""
    from brain.pages import parse

    home = fetcher.get(f"https://{org.org_domain}/")
    if not home.ok:
        return False
    parsed = parse(home.html, home.url)
    link = next((h for h, t in parsed.nav_links
                 if registrable_domain(h) == org.org_domain and (_PORTFOLIO_LINK.search(urlsplit(h).path) or
                                                                  _PORTFOLIO_LINK.search(t or ""))), None)
    page = fetcher.get(link) if link else home
    if not page.ok:
        return False
    pp = parse(page.html, page.url)
    low = pp.text.lower()
    matched = [t for t in dict.fromkeys(topic_terms) if t in low]
    companies = {registrable_domain(h) for h, _ in pp.links if registrable_domain(h) != org.org_domain}
    if not matched and len(companies) < 5:
        return False
    org.signals["portfolio_fit"] = min(1.0, 0.3 * len(matched)) if matched else 0.2
    org.signals["portfolio_size"] = float(len(companies))
    org.evidence.append(Evidence(kind="page", url=page.url,
                                 text=squash(f"Portfolio page lists ~{len(companies)} companies"
                                             + (f"; mentions {', '.join(matched[:4])}" if matched else ""), 250)))
    return True


def discover_portfolio(run) -> Tuple[List[Contact], Dict]:
    """Investors found by search whose portfolio pages mention the campaign's topics."""
    from brain.orgs import discover_orgs

    spec = run.spec
    place = (spec.regions or [""])[0] or (spec.countries[0] if spec.countries else "")
    queries = [squash(f"{t} venture capital fund {place} portfolio") for t in spec.topics[:3]] + \
              [squash(f"pre-seed seed investors {t} {place}") for t in spec.topics[:2]]
    queries = [q for q in queries if not run.ctx.tried(q)]
    orgs, stats = discover_orgs(spec, run.playbook, run.deps.search, run.deps.fetcher, run.deps.llm,
                                run.deps.settings, planner=run.planner, queries=queries, budget=run.budget)
    terms = [t.lower() for t in spec.topics + spec.expanded_terms]
    checked = 0
    for o in orgs[:20]:
        if o.org_domain and portfolio_evidence(run.deps.fetcher, o, terms):
            checked += 1
            o.sources = list(dict.fromkeys(o.sources + ["portfolio"]))
    stats["portfolios_read"] = checked
    return orgs, stats


# ── podcasts: iTunes directory + RSS ─────────────────────────────────────────

def parse_feed(xml: str) -> Dict[str, object]:
    """Kept for callers of the old name; the podcasts source is a manifest now (brain/discovery/sources/podcasts.json)."""
    from brain.discovery.adapters.rss import parse_feed as generic

    return generic(xml)


def _recent(date_text: str, days: int = 90) -> bool:
    from email.utils import parsedate_to_datetime

    try:
        d = parsedate_to_datetime(date_text)
    except (TypeError, ValueError):
        return False
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - d).days <= days


# ── expansion from known-good prospects ──────────────────────────────────────

def discover_lookalike(run) -> Tuple[List[Contact], Dict]:
    from brain.orgs import discover_orgs

    names = [(w.org or w.name) for w in run.winners[:4]] or list(run.ctx.seed_names[:4])
    queries = run.planner.lookalikes(names)
    if not queries:
        return [], {"queries": 0}
    found, stats = discover_orgs(run.spec, run.playbook, run.deps.search, run.deps.fetcher, run.deps.llm,
                                 run.deps.settings, planner=run.planner, queries=queries[:6], budget=run.budget)
    for c in found:
        c.sources = list(dict.fromkeys(c.sources + ["lookalike"]))
    return found, stats


def discover_coauthors(run) -> Tuple[List[Contact], Dict]:
    source = run.deps.openalex
    if source is None or not hasattr(source, "coauthors"):
        return [], {"skipped": "no OpenAlex source"}
    seeds = [(k.split(":", 1)[1], w.name) for w in run.winners for k in w.keys if k.startswith("openalex:")][:5]
    if not seeds:
        return [], {"seeds": 0}
    found = source.coauthors(seeds, run.spec)
    return found, {"seeds": len(seeds), "people": len(found)}


def discover_memory(run) -> Tuple[List[Contact], Dict]:
    from brain.fit import profile_for

    store = run.deps.store
    if not getattr(run.deps.settings, "use_memory", True) or not hasattr(store, "known_prospects"):
        return [], {}
    found = store.known_prospects(run.spec, profile_for(run.spec, run.playbook), limit=run.spec.target_count * 3)
    return found, {"known": len(found)}
