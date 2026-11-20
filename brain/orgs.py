"""
Organisation-first discovery and enrichment, for every campaign outside academia.

Sponsors, clinics, startups and investment funds are found as *organisations* first; who to email
inside them is decided only after reading their own website:

    search -> organisation candidates            (their own sites, or names pulled from list pages)
           -> cheap ranking
           -> read each site (home, contact, about, team pages)
           -> a named person whose role fits, else the right shared inbox (partnerships@, info@ ...)

The LLM is used only to pull organisation names out of list pages ("Top 20 edtech startups"),
which heuristics cannot do reliably. Everything else is page fetching and parsing.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from brain.discover import rank_hits
from brain.emailfinder import _patterns, extract_emails, infer_domain_patterns, is_role_address, name_match
from brain.fetch import Fetcher
from brain.llm import LLM, LLMError
from brain.models import CampaignSpec, Contact, EmailCandidate, Evidence
from brain.packs import Pack
from brain.pages import ParsedPage, classify, directory_contacts, parse, profile_contact
from brain.settings import Settings
from brain.sources.websearch import Hit, WebSearch
from brain.textutil import (
    FREE_MAIL, clean_name, country_from_domain, host_of, looks_like_person_name, region_terms, registrable_domain,
    squash, to_country_code,
)

log = logging.getLogger("brain.orgs")

# Sites that list or describe other organisations. Never a target themselves; read for names.
AGGREGATORS = (
    "wellfound.com", "angel.co", "ycombinator.com", "crunchbase.com", "tracxn.com", "f6s.com", "cbinsights.com",
    "practo.com", "justdial.com", "lybrate.com", "sulekha.com", "indiamart.com", "clinicspots.com", "bookimed.com",
    "glassdoor.", "naukri.com", "instahyre.com", "placementindia.com", "monster", "indeed.", "cutshort.io",
    "vcsheet.com", "failory.com", "angelmatch.io", "openvc.app", "startupmag.co.uk", "inc42.com", "yourstory.com",
    "entrackr.com", "clutch.co", "goodfirms.co", "g2.com", "capterra.", "yelp.", "tripadvisor.", "zaubacorp.com",
    "tofler.in", "builtin.com", "producthunt.com", "medium.com", "substack.com", "quora.com", "reddit.com",
    "linkedin.com", "facebook.com", "instagram.com", "x.com", "twitter.com", "youtube.com", "wikipedia.org",
    "blogspot.", "wordpress.com", "eventbrite.", "allevents.in", "townscript.com", "unstop.com", "devfolio.co",
)
_LIST_TITLE = re.compile(
    r"\b(top|best|list|leading|directory|ranking|\d{1,3}\+?)\b.*\b(compan(y|ies)|startups?|firms|funds|investors|"
    r"angels|vcs|clinics|hospitals|brands|agencies|sponsors|organi[sz]ations|businesses|platforms|tools)\b", re.I)

_PAGE_HINTS = [
    (5, re.compile(r"contact|reach-us|get-in-touch|enquir", re.I)),
    (4, re.compile(r"team|people|leadership|founders|our-story|management|partners|doctors|faculty|staff", re.I)),
    (3, re.compile(r"about|company|who-we-are", re.I)),
    (2, re.compile(r"sponsor|partner(ship)?s?|invest|portfolio|careers|jobs|join", re.I)),
]
_AVOID = re.compile(r"blog|news|press|privacy|terms|cookie|login|signup|cart|shop|product/|\.pdf|wp-content|"
                    r"tag/|category/|author/|page/\d", re.I)

DEFAULT_INBOXES = [
    "partnerships", "partnership", "partners", "partner", "sponsorship", "sponsorships", "sponsor", "brand",
    "marketing", "collab", "collaborations", "business", "bd", "biz", "pitch", "pitches", "deals", "invest",
    "investments", "founders", "hello", "hi", "contact", "contactus", "info", "enquiry", "enquiries", "inquiry",
    "careers", "jobs", "hiring", "talent", "hr", "recruitment", "appointments", "appointment", "booking", "reception",
    "clinic", "office", "admin", "team", "mail",
]
NEVER_INBOXES = {"noreply", "no-reply", "donotreply", "do-not-reply", "webmaster", "postmaster", "hostmaster", "abuse",
                 "privacy", "security", "legal", "billing", "accounts", "invoice", "support", "help", "helpdesk",
                 "newsletter", "unsubscribe", "dpo", "grievance", "compliance", "example", "test", "user", "name",
                 "email", "your", "yourname", "sentry", "bounce", "token", "api", "dev", "demo", "sample",
                 "foo", "bar", "john", "jane", "johndoe", "janedoe", "someone", "me", "abc", "xyz", "username"}

ORG_LIST_SYSTEM = """You read a web page and list the organisations on it that match the TARGET.
Return JSON: {"organisations": [{"name": "<name exactly as written on the page>",
"website": "<that organisation's own site: copy one URL from LINKS, or null>",
"about": "<one sentence from the page about them, or null>", "location": "<city/region if stated, or null>"}]}
Only organisations named on the page; never the page's publisher; at most 30. Prefer ones that fit the TARGET.
If the page names no organisations, return {"organisations": []}."""


def is_aggregator(domain: str) -> bool:
    return any(a in domain for a in AGGREGATORS)


_GENERIC_TITLE = re.compile(
    r"^(home|homepage|welcome|contact( us)?|about( us)?|official (web)?site|sponsorships?|partnerships?|events?|"
    r"careers?|jobs|blog|news|services|solutions|products|pricing|team|our team|portfolio|investors?)$", re.I)


def _plausible_org_name(part: str) -> bool:
    return (2 <= len(part) <= 50 and "?" not in part and len(part.split()) <= 6
            and not _GENERIC_TITLE.match(part) and not part.lower().startswith(("how ", "what ", "why ", "top ", "best ")))


def _org_name_from_title(title: str, domain: str) -> str:
    parts = [p.strip() for p in re.split(r"\s*[|·–—:]\s*|\s+-\s+", title or "") if p.strip()]
    parts = [p for p in parts if _plausible_org_name(p)]
    label = domain.split(".")[0]
    # Prefer the part that resembles the domain ("Smile Dental | Best dentist in X" on smiledental.in)
    for p in parts:
        squeezed = re.sub(r"[^a-z]", "", p.lower())
        if squeezed and (squeezed[:6] in label or label[:6] in squeezed):
            return squash(p, 80)
    return label.replace("-", " ").title()  # the domain is a safer name than a page heading


def _org_contact(name: str, domain: Optional[str], url: Optional[str], about: str = "", location: str = "",
                 source: str = "web") -> Contact:
    c = Contact(kind="org", name=name, org=name, org_domain=domain, sources=[source],
                country=country_from_domain(domain) if domain else None, location_text=location or None)
    if url:
        c.add_url(url)
    if about:
        c.evidence.append(Evidence(kind="profile", text=squash(about, 300), url=url))
    return c


# ── discovery ────────────────────────────────────────────────────────────────

def _orgs_from_list_page(html: str, url: str, spec: CampaignSpec, llm: LLM) -> List[Contact]:
    page = parse(html, url)
    own = registrable_domain(url)
    links: Dict[str, str] = {}
    for href, text in page.nav_links + page.links:
        dom = registrable_domain(href)
        if dom != own and not is_aggregator(dom) and dom not in links.values():
            links[href] = dom
    link_lines = "\n".join(f"- {u}" for u in list(links)[:150])
    target = f"{spec.persona.description} | topics: {', '.join(spec.topics[:5])} | where: {', '.join(spec.regions + spec.countries) or 'anywhere'}"
    from brain.safety import clean_text, guard

    user = f"TARGET: {target}\nPAGE: {url}\n\nTEXT:\n{clean_text(page.text[:6000])}\n\nLINKS:\n{link_lines or '(none)'}"
    data = llm.json(system=guard(ORG_LIST_SYSTEM), user=user, purpose="extract_orgs", tier="fast", max_tokens=2500)
    text_low = page.text.lower()
    out: List[Contact] = []
    for o in (data.get("organisations") or [])[:30] if isinstance(data, dict) else []:
        if not isinstance(o, dict):
            continue
        name = squash(str(o.get("name") or ""), 80)
        if len(name) < 2 or name.lower() not in text_low:
            continue  # must literally be on the page
        site = str(o.get("website") or "").strip()
        site = site if site in links else ""
        dom = registrable_domain(site) if site else None
        out.append(_org_contact(name, dom, site or None, str(o.get("about") or ""), str(o.get("location") or ""),
                                source="list"))
        out[-1].evidence.append(Evidence(kind="page", text=f"Listed on {squash(page.title, 120)}", url=url))
    return out


def discover_orgs(spec: CampaignSpec, pack: Pack, search: WebSearch, fetcher: Fetcher, llm: Optional[LLM],
                  settings: Settings, planner: Optional["SearchPlanner"] = None,
                  max_list_pages: int = 8, queries: Optional[List[str]] = None,
                  budget: Optional[int] = None) -> Tuple[List[Contact], Dict[str, int]]:
    """Organisations from web search. Each query's results are triaged (own site / list / event / article),
    processed at once (own sites kept, list and event pages read for names) and scored against the prospect
    profile, so the planner knows how many QUALIFIED organisations every query yielded and stops as soon as
    there are enough. `queries` runs exactly those (lookalike / feedback rounds); `budget` caps search calls."""
    from brain.channels import _place_terms, orgs_from_event_page
    from brain.context import CampaignContext
    from brain.fit import profile_for, promising
    from brain.searchplan import SearchPlanner
    from brain.triage import triage

    stats = {"queries": 0, "hits": 0, "org_sites": 0, "list_pages": 0, "orgs_from_lists": 0, "event_pages": 0,
             "orgs_from_events": 0, "articles_skipped": 0, "irrelevant_skipped": 0, "qualified": 0}
    if not search.available:
        return [], stats
    planner = planner or SearchPlanner(spec, pack, CampaignContext(), llm)
    prof = profile_for(spec, pack)
    orgs: Dict[str, Contact] = {}
    good: set = set()
    seen_urls: set = set()
    start_calls = search.calls
    triage_llm = llm if getattr(settings, "triage_llm", True) else None

    def add(c: Contact) -> Tuple[bool, bool]:
        key = c.org_domain or c.name.lower()
        if key in orgs:
            orgs[key].evidence += [e for e in c.evidence if e.text not in {x.text for x in orgs[key].evidence}]
            for k, v in c.signals.items():
                orgs[key].signals[k] = max(orgs[key].signals.get(k, 0.0), v)
            if key not in good and promising(orgs[key], prof):
                good.add(key)
            return False, False
        orgs[key] = c
        ok = promising(c, prof)
        if ok:
            good.add(key)
        return True, ok

    def execute(query: str) -> Tuple[int, int, int]:
        stats["queries"] += 1
        hits = [h for h in search.search(query, n=10) if h.url not in seen_urls]
        seen_urls.update(h.url for h in hits)
        stats["hits"] += len(hits)
        new = qualified = 0
        lists: List[Hit] = []
        events: List[Hit] = []
        ranked = [h for _, h in rank_hits(hits, spec, pack)]
        for h, kind in triage(ranked, spec, is_aggregator, triage_llm, stats):
            if kind == "list":
                lists.append(h)
                continue
            if kind == "event":
                events.append(h)
                continue
            if kind in ("article", "irrelevant"):
                stats[f"{kind}s_skipped" if kind == "article" else "irrelevant_skipped"] += 1
                continue
            dom = registrable_domain(h.url)
            if is_aggregator(dom):
                continue
            home = f"{urlsplit(h.url).scheme or 'https'}://{host_of(h.url)}/"
            c = _org_contact(_org_name_from_title(h.title, dom), dom, home, h.snippet)
            c.add_url(h.url)
            is_new, ok = add(c)
            new += is_new
            qualified += is_new and ok
            stats["org_sites"] += is_new
        for h in events[:2]:  # sponsors / partners of a similar event: proof of past sponsorship, no LLM
            page = fetcher.get(h.url)
            if not page.ok:
                continue
            stats["event_pages"] += 1
            for c in orgs_from_event_page(page.html, page.url, h.title, place_terms=_place_terms(spec)):
                is_new, ok = add(c)
                new += is_new
                qualified += is_new and ok
                stats["orgs_from_events"] += is_new
        if llm is not None and llm.available:
            for h in lists[:2]:  # the two most relevant list pages per query
                if stats["list_pages"] >= max_list_pages:
                    break
                page = fetcher.get(h.url)
                if not page.ok:
                    continue
                try:
                    found = _orgs_from_list_page(page.html, page.url, spec, llm)
                except LLMError as e:
                    log.warning("list extraction failed for %s: %s", h.url, e)
                    break
                stats["list_pages"] += 1
                for c in found:
                    is_new, ok = add(c)
                    new += is_new
                    qualified += is_new and ok
                    stats["orgs_from_lists"] += is_new
        return len(hits), new, qualified

    def budget_left() -> int:
        left = search.remaining - 2
        if budget is not None:
            left = min(left, budget - (search.calls - start_calls))
        return left

    planner.run(execute, have=lambda: len(good), budget_left=budget_left, kind="search", queries=queries)
    stats["qualified"] = len(good)
    return list(orgs.values()), stats


def discover_places(spec: CampaignSpec, pack: Pack, search: WebSearch, planner: "SearchPlanner",
                    gl: str = "", budget: Optional[int] = None) -> Tuple[List[Contact], Dict[str, int]]:
    """Local businesses from Google Maps: one call returns ~20 businesses with website and address."""
    stats = {"queries": 0, "places": 0, "with_website": 0}
    orgs: Dict[str, Contact] = {}

    def execute(query: str) -> Tuple[int, int]:
        stats["queries"] += 1
        found = search.places(query, gl=gl)
        stats["places"] += len(found)
        new = 0
        for p in found:
            if not p.website:
                continue  # without a website there is nothing to read and no address to find
            dom = registrable_domain(p.website)
            if not dom or is_aggregator(dom) or dom in orgs:
                continue
            about = ", ".join(x for x in [p.category, f"rated {p.rating} ({p.reviews} reviews)" if p.reviews else ""] if x)
            c = _org_contact(p.name, dom, p.website, about or p.name, location=p.address, source="places")
            if p.reviews:
                c.signals["reviews"] = float(p.reviews)
            c.signals["local_business"] = 1.0  # a Maps listing in the right category is direct proof
            orgs[dom] = c
            new += 1
        stats["with_website"] = len(orgs)
        return len(found), new

    start = search.calls

    def budget_left() -> int:
        left = search.remaining - 2
        return min(left, budget - (search.calls - start)) if budget is not None else left

    planner.run(execute, have=lambda: len(orgs), budget_left=budget_left, kind="places", first_round_max=6)
    return list(orgs.values()), stats


# ── enrichment ───────────────────────────────────────────────────────────────

# Role words hiding inside one-token local parts ("mediarelations@", "factcheck@", "pr@"): never an owner's address.
_FUNCTION_IN_LOCAL = re.compile(r"^(pr|ir|cs)$|media|relation|press|news|fact|check|support|sales|career|recruit|"
                                r"admin|office|contact|enquir|inquir|market|partner|invest|legal|help|service|feedback|"
                                r"desk|editor|webmaster|booking|order|account|billing|info", re.I)

GENERIC_INBOXES = {"hello", "hi", "contact", "contactus", "info", "enquiry", "enquiries", "inquiry", "team", "mail",
                   "office", "admin", "business", "biz"}
# Regional inboxes ("info-latam@") reach the wrong team unless the campaign targets that region.
_REGION_TOKENS = {
    "latam": {"BR", "MX", "AR", "CL", "CO", "PE"}, "emea": {"GB", "DE", "FR", "NL", "AE", "ZA", "ES", "IT", "SE", "IE"},
    "apac": {"IN", "SG", "AU", "JP", "ID", "MY", "PH", "VN", "TH", "KR", "NZ", "HK"}, "americas": {"US", "CA", "BR", "MX"},
    "mena": {"AE", "SA", "EG", "QA"}, "anz": {"AU", "NZ"}, "eu": {"DE", "FR", "NL", "ES", "IT", "IE", "SE"},
    "europe": {"DE", "FR", "NL", "ES", "IT", "IE", "SE", "GB"}, "us": {"US"}, "usa": {"US"}, "uk": {"GB"},
    "india": {"IN"}, "sg": {"SG"}, "de": {"DE"}, "fr": {"FR"}, "br": {"BR"}, "mx": {"MX"}, "jp": {"JP"}, "au": {"AU"},
    "ca": {"CA"},
}


def _inbox_rank(address: str, preferred: List[str], countries: List[str] = ()) -> Optional[int]:
    """Lower is better; None means this address must not be used. With `preferred` set, a shared inbox for another
    function (careers@ for a sponsorship ask) is refused; generic ones (hello@, info@) stay acceptable."""
    tokens = [t for t in re.split(r"[._\-+]", address.split("@")[0].lower()) if t]
    if not tokens or any(t in NEVER_INBOXES for t in tokens) and not any(t in preferred for t in tokens):
        return None
    if preferred and not any(t in preferred or t in GENERIC_INBOXES for t in tokens) and \
            any(t in DEFAULT_INBOXES for t in tokens):
        return None
    if countries and any(t in _REGION_TOKENS and not (_REGION_TOKENS[t] & set(countries)) for t in tokens):
        return None
    order = [p.lower() for p in preferred] + [d for d in DEFAULT_INBOXES if d not in preferred]
    ranks = [order.index(t) for t in tokens if t in order]
    return min(ranks) if ranks else None


def _role_fits(title: Optional[str], roles: List[str]) -> bool:
    if not title or not roles:
        return False
    t = title.lower()
    return any(r.lower() in t or t in r.lower() for r in roles)


# Pages that are about an organisation but are not its own website (a result for "<name> official website" can be one).
NOT_OWN_SITE = {"google.com", "facebook.com", "instagram.com", "linkedin.com", "youtube.com", "twitter.com", "x.com",
                "wikipedia.org", "justdial.com", "indiamart.com", "tripadvisor.com", "yelp.com", "pinterest.com",
                "crunchbase.com", "zaubacorp.com", "tofler.in"}


class OrgEnricher:
    def __init__(self, fetcher: Fetcher, search: WebSearch, settings: Settings, find_people: bool = False,
                 max_people_searches: int = 6, store=None, accept_prior: bool = False):
        self.store, self.accept_prior = store, accept_prior  # cross-campaign email conventions (reach/patterns.py)
        self.fetcher, self.search, self.settings = fetcher, search, settings
        self.find_people = find_people
        self.max_people_searches = max_people_searches
        self._people_searches = 0
        self._resolves = 0
        self._snippet_searches = 0
        self.max_resolves = 8  # raised to follow the campaign size in enrich_all
        self._lock = threading.Lock()
        self._pending: Dict[int, tuple] = {}  # organisations waiting for a decision-maker search (enrich_all)
        self.stats = {"orgs_read": 0, "sites_resolved": 0, "person_contacts": 0, "inbox_contacts": 0,
                      "site_contacts": 0, "no_contact": 0, "decision_maker_searches": 0, "decision_makers_found": 0,
                      "decision_makers_emailed": 0}

    def _find_decision_maker(self, org: Contact, roles: List[str], site_people: List[Contact]) -> Optional[Contact]:
        """Search for the person in the right role at this organisation. Their name and title are recorded on the
        organisation (so the email can address them) even when only the shared inbox is known; a personal
        contact is returned only when the address follows a convention confirmed on the organisation's site."""
        # A contact-data provider, when configured, goes first: it names the person in the role AND holds an address
        # it has checked, for one provider call instead of a web search plus a guessed address.
        from brain.discovery.reach.providers import hunter_available, hunter_find, hunter_person

        if hunter_available():
            org.flags.append("hunter_tried")  # the later fallback must not pay for the same lookup again
            paid = hunter_person(self.fetcher, org, roles)
            if paid is not None:
                self.stats["hunter_contacts"] = self.stats.get("hunter_contacts", 0) + 1
                org.decision_maker = {"name": paid.name, "title": paid.title or "", "source": "hunter", "confidence": 1.0}
                return paid
        with self._lock:
            if self._people_searches >= self.max_people_searches or self.search.remaining <= 4:
                return None
            self._people_searches += 1
            self.stats["decision_maker_searches"] += 1
        top = [r for r in roles[:3] if r]
        query = f'"{org.name}" ' + " OR ".join(f'"{r}"' for r in top)
        hits = self.search.search(query, n=8, raw=True)
        person = people_from_serp_scored(hits, org.name, top)
        if not person:
            return None
        name, title, url, conf = person
        self.stats["decision_makers_found"] += 1
        org.decision_maker = {"name": name, "title": title, "source": url, "confidence": conf}
        patterns = infer_domain_patterns(site_people, min_support=2)
        hit = next(((d, pn) for d, pn in patterns.items() if registrable_domain(d) == org.org_domain), None)
        c = Contact(kind="person", name=name, title=title, org=org.name, org_domain=org.org_domain,
                    country=org.country, location_text=org.location_text, urls=[url] + org.urls,
                    evidence=list(org.evidence), bio=org.bio, sources=org.sources + ["decision_maker"],
                    flags=org.flags + ["email_from_pattern"], signals=dict(org.signals))
        found = hunter_find(self.fetcher, name, title, org) if hunter_available() else None
        if found is not None:  # the provider knows this person's address: no guessing from a convention
            c.add_email(found)
            c.flags = [f for f in c.flags if f != "email_from_pattern"]
            self.stats["hunter_found"] = self.stats.get("hunter_found", 0) + 1
            self.stats["decision_makers_emailed"] += 1
            return c
        if hit:
            domain, (pname, support) = hit
            local = _patterns(name).get(pname)
            if not local:
                return None
            c.add_email(EmailCandidate(address=f"{local}@{domain}", confidence=0.7 if support >= 3 else 0.62,
                                       method="pattern_confirmed", source_url=url))
        else:
            # No convention on the site: the one confirmed in earlier campaigns, or (if the playbook accepts
            # flagged guesses) the usual one for this kind of organisation.
            from brain.discovery.reach.patterns import guess

            g = guess(c, self.store, self.accept_prior, domain=org.org_domain)
            if g is None:
                return None
            g.source_url = url
            c.add_email(g)
            c.flags.append("email_guessed")
        self.stats["decision_makers_emailed"] += 1
        return c

    def _resolve_site(self, org: Contact, spec: CampaignSpec) -> None:
        """Find the official website of an organisation known only by name (one search, capped per campaign).
        A hit is accepted when the domain carries a word of the name, or, for acronym domains (unal.edu.co for
        "Universidad Nacional de Colombia"), when the result's title or snippet names the organisation in full."""
        with self._lock:
            if self._resolves >= self.max_resolves or not _plausible_org_name(org.name):
                return
            self._resolves += 1
        place = (spec.regions or [""])[0]
        hits = self.search.search(f'"{org.name}" {place} official website', n=5)
        tokens = [t for t in re.findall(r"[a-z0-9]+", org.name.lower()) if len(t) > 2]
        full = " ".join(re.findall(r"[a-z0-9]+", org.name.lower()))
        for rank, h in enumerate(hits):
            dom = registrable_domain(h.url)
            label = dom.split(".")[0]
            if is_aggregator(domain=dom) or dom in NOT_OWN_SITE:
                continue
            said = " ".join(re.findall(r"[a-z0-9]+", f"{h.title} {h.snippet}".lower()))
            if any(t in label for t in tokens) or label in "".join(tokens) or (rank < 2 and len(full) > 8 and full in said):
                org.org_domain = dom
                org.country = org.country or country_from_domain(dom)
                org.add_url(f"https://{host_of(h.url)}/")
                self.stats["sites_resolved"] += 1
                return

    def read_home(self, org: Contact) -> Optional[Tuple["ParsedPage", str]]:
        """Fetch and read an organisation's homepage: real name, description, hiring signal. Cached, so the
        light read before ranking and the full enrichment after it pay for the page once."""
        home = next((u for u in org.urls if registrable_domain(u) == org.org_domain and urlsplit(u).path in ("", "/")),
                    f"https://{org.org_domain}/")
        page = self.fetcher.get(home)
        if not page.ok and "//www." not in home:
            page = self.fetcher.get(home.replace("://", "://www.", 1))
        if not page.ok:
            return None
        parsed = parse(page.html, page.url)
        soup = parsed.soup
        meta = None
        if soup is not None:
            for attrs in ({"property": "og:site_name"}, {"name": "application-name"}):
                tag = soup.find("meta", attrs=attrs)
                if tag and tag.get("content") and _plausible_org_name(tag["content"].strip()):
                    org.name = org.org = squash(tag["content"], 80)
                    break
            else:
                org.name = org.org = _org_name_from_title(parsed.title, org.org_domain)
            for attrs in ({"name": "description"}, {"property": "og:description"}):
                tag = soup.find("meta", attrs=attrs)
                if tag and tag.get("content") and len(tag["content"]) > 40:
                    meta = squash(tag["content"], 300)
                    break
        description = meta or next((squash(p) for p in parsed.text.split("\n") if len(p.strip()) > 80), "")
        if description and description[:300] not in {e.text for e in org.evidence}:
            org.bio = description[:400]
            org.evidence.insert(0, Evidence(kind="profile", text=description[:300], url=page.url))
        hiring = hiring_signal(parsed)
        if hiring:
            org.signals["hiring"] = max(org.signals.get("hiring", 0.0), hiring)
        return parsed, page.url

    def light_read(self, orgs: List[Contact], spec: CampaignSpec, limit: int) -> int:
        """Homepage-only read of the first `limit` organisations, before pre-ranking, so ranking sees what the
        organisation says about itself instead of a search snippet. Returns how many were read."""
        todo = [o for o in orgs if o.kind == "org" and o.org_domain and "light_read" not in o.flags][:limit]

        def one(o: Contact) -> bool:
            try:
                got = self.read_home(o)
            except Exception as e:
                log.warning("light read failed for %s: %s", o.org_domain, e)
                return False
            o.flags.append("light_read")
            if got:
                low = got[0].text.lower()
                places = [r for r in spec.regions if any(t in low for t in region_terms(r))]
                if places and not o.location_text:
                    o.location_text = ", ".join(places)
            return bool(got)

        with ThreadPoolExecutor(max_workers=max(1, self.settings.fetch_workers)) as ex:
            return sum(ex.map(one, todo))

    def enrich(self, org: Contact, spec: CampaignSpec, roles: List[str], defer_dm: bool = False) -> Contact:
        """Read the organisation's site and choose who to write to. With `defer_dm`, an organisation that needs a
        decision-maker search is parked (see enrich_all) instead of searching right away, so the capped searches
        go to the best-ranked organisations in order rather than to whichever thread gets there first."""
        if "channel_verified" in org.flags and org.emails:
            return org  # the channel already gave a published address (e.g. a podcast feed): nothing to look up
        if org.org_domain in NOT_OWN_SITE:  # a map or social page is not the organisation's own site
            org.add_url(f"https://{org.org_domain}/")
            org.org_domain = None
        if not org.org_domain:
            self._resolve_site(org, spec)
        if not org.org_domain:
            org.flags.append("no_website")
            return org
        got = self.read_home(org)
        if not got:
            org.flags.append("site_unreachable")
            self._from_snippets(org, spec)  # the site will not load; a search result may still quote its address
            return org
        self.stats["orgs_read"] += 1
        parsed, page_url = got
        pages = [parsed]
        # Follow the few internal pages most likely to name people or give contact details.
        scored: Dict[str, int] = {}
        for href, text in parsed.nav_links:
            if registrable_domain(href) != org.org_domain or _AVOID.search(href) or href.rstrip("/") == page_url.rstrip("/"):
                continue
            score = max((w for w, rx in _PAGE_HINTS if rx.search(href) or rx.search(text)), default=0)
            if score:
                scored[href] = max(scored.get(href, 0), score)
        for extra in self.fetcher.get_many(sorted(scored, key=scored.get, reverse=True)[:4]):
            if extra.ok:
                pages.append(parse(extra.html, extra.url))

        emails: Dict[str, str] = {}
        people: List[Contact] = []
        all_text = []
        for p in pages:
            all_text.append(p.text)
            for e in p.emails:
                emails.setdefault(e, p.url)
            kind = classify(p)
            if kind == "directory":
                people += directory_contacts(p)
            elif kind == "profile":
                pc = profile_contact(p)
                if pc and pc.name.lower() not in org.name.lower():
                    people.append(pc)
            # team cards often show "Name — Title" without links: catch titled names in text
        text = "\n".join(all_text)

        # Location: what the site itself says about where it is.
        low = text.lower()
        found_places = [r for r in spec.regions if any(t in low for t in region_terms(r))]
        for code in spec.countries:
            names = [k for k, v in _country_names().items() if v == code]
            if any(n in low for n in names):
                org.country = org.country or code
        if found_places:
            org.location_text = ", ".join(found_places)
        org.flags.append("enriched")

        # 1. A named person whose role fits, with an address that is clearly theirs.
        best_person = None
        for person in people:
            good = [e for e in person.emails if e.confidence >= self.settings.min_email_confidence
                    and name_match(person.name, e.address) > 0]
            if not good:
                own = [a for a in emails if name_match(person.name, a) == 1.0
                       and registrable_domain(a.split("@")[1]) in (org.org_domain, *FREE_MAIL)]
                good = [EmailCandidate(address=a, confidence=0.85, method="text", source_url=emails[a]) for a in own[:1]]
            if good and (best_person is None or _role_fits(person.title, roles)):
                best_person = (person, good)
                if _role_fits(person.title, roles):
                    break
        if best_person and (_role_fits(best_person[0].title, roles) or not self.find_people):
            return self._person_contact(org, best_person)

        # 2. Nobody in the right role on the site: look for the decision-maker by role (one search). A personal
        #    address is used only when the organisation's own convention is confirmed by people on its site.
        if self.find_people and roles:
            if defer_dm:
                with self._lock:
                    self._pending[id(org)] = (people, emails, best_person)
                return org
            dm = self._find_decision_maker(org, roles, people)
            if dm is not None:
                return dm
        return self._fallback(org, spec, emails, best_person, roles)

    def _person_contact(self, org: Contact, best_person) -> Contact:
        person, good = best_person
        c = Contact(kind="person", name=person.name, title=person.title, org=org.name, org_domain=org.org_domain,
                    country=org.country, location_text=org.location_text, urls=person.urls + org.urls,
                    evidence=org.evidence + person.evidence, bio=org.bio, sources=org.sources, flags=org.flags,
                    signals=dict(org.signals))
        for e in good:
            c.add_email(e)
        self.stats["person_contacts"] += 1
        return c

    def _fallback(self, org: Contact, spec: CampaignSpec, emails: Dict[str, str], best_person,
                  roles: Optional[List[str]] = None) -> Contact:
        """No decision-maker with an address: someone reachable, else a paid provider's personal address (only with
        its key), else the right shared inbox, else the owner's."""
        # Someone reachable at the organisation, even if not in the ideal role — only when the address is clearly
        # theirs (a page title such as "Partner Spotlight" can look like a name and half-match partners@).
        if best_person and any(name_match(best_person[0].name, e.address) == 1.0 for e in best_person[1]):
            return self._person_contact(org, best_person)
        if self.find_people and "hunter_tried" not in org.flags:
            from brain.discovery.reach.providers import hunter_person

            paid = hunter_person(self.fetcher, org, roles or [])
            if paid is not None:
                self.stats["hunter_contacts"] = self.stats.get("hunter_contacts", 0) + 1
                return paid

        # 3. The organisation's shared inbox, chosen to suit the ask (partnerships@ for sponsorship, ...).
        preferred = [p.lower() for p in spec.preferred_inboxes]
        candidates = []
        for addr, src in emails.items():
            dom = registrable_domain(addr.split("@")[1])
            if dom != org.org_domain:
                continue
            rank = _inbox_rank(addr, preferred, spec.countries)
            if rank is not None and is_role_address(addr) or rank is not None and rank < len(preferred):
                candidates.append((rank, addr, src))
        if candidates:
            _, addr, src = min(candidates)
            org.add_email(EmailCandidate(address=addr, confidence=0.7, method="inbox", source_url=src))
            self.stats["inbox_contacts"] += 1
            return org

        # 4. Small organisations often publish one personal-looking address (the owner's): use it if it is the only one.
        personal = [a for a in emails if not is_role_address(a) and not _FUNCTION_IN_LOCAL.search(a.split("@")[0])
                    and _inbox_rank(a, preferred, spec.countries) is None
                    and registrable_domain(a.split("@")[1]) in (org.org_domain, *FREE_MAIL)]
        if len(personal) == 1:
            org.add_email(EmailCandidate(address=personal[0], confidence=0.65, method="site", source_url=emails[personal[0]]))
            self.stats["site_contacts"] += 1
            return org
        if self._from_snippets(org, spec):
            self.stats["inbox_contacts"] += 1
            return org
        self.stats["no_contact"] += 1
        return org

    def _from_snippets(self, org: Contact, spec: CampaignSpec) -> bool:
        """One search for the organisation's own address, read from result titles and snippets only (no page fetch).
        Used when the site would not load or published nothing: 84% of the valid candidates lost in the stress runs had
        no address at all. Only addresses on the organisation's own domain are taken, as an inbox (not a person)."""
        from brain.emailfinder import extract_emails

        if not org.org_domain:
            return False
        with self._lock:
            if self._snippet_searches >= self.max_resolves:
                return False
            self._snippet_searches += 1
        hits = self.search.search(f'"{org.name}" {org.org_domain} email contact', n=6)
        found: Dict[str, str] = {}
        for h in hits:
            for addr in extract_emails(f"{h.title} {h.snippet}"):
                if registrable_domain(addr.split("@")[1]) == org.org_domain:
                    found.setdefault(addr.lower(), h.url)
        if not found:
            return False
        preferred = [p.lower() for p in spec.preferred_inboxes]
        ranked = sorted(found, key=lambda a: (_inbox_rank(a, preferred, spec.countries) is None,
                                              _inbox_rank(a, preferred, spec.countries) or 0))
        before = len(org.emails)
        org.add_email(EmailCandidate(address=ranked[0], confidence=0.6, method="inbox", source_url=found[ranked[0]]))
        if len(org.emails) > before:
            org.flags.append("email_from_snippet")
            self.stats["snippet_contacts"] = self.stats.get("snippet_contacts", 0) + 1
            return True
        return False

    def enrich_all(self, orgs: List[Contact], spec: CampaignSpec, pack: Pack) -> List[Contact]:
        roles = spec.profile.decision_roles or spec.persona.roles or pack.default_roles
        # One search per organisation known only by name. A fixed 8 left 106 of ~300 valid, named candidates in the
        # stress runs with no website and so no address; the cap now follows the size of the campaign.
        self.max_resolves = max(self.max_resolves, min(80, 4 * spec.target_count))

        def one(o: Contact) -> Contact:
            try:
                return self.enrich(o, spec, roles, defer_dm=True)
            except Exception as e:  # one broken site must not abort the batch
                log.warning("org enrichment failed for %s: %s", o.name, e)
                o.flags.append("enrich_error")
                return o

        # Sites are read in parallel; the capped decision-maker searches then run in rank order (deterministic).
        with ThreadPoolExecutor(max_workers=max(1, self.settings.fetch_workers)) as ex:
            out = list(ex.map(one, orgs))
        for i, o in enumerate(out):
            parked = self._pending.pop(id(o), None)
            if parked is None:
                continue
            people, emails, best_person = parked
            try:
                dm = self._find_decision_maker(o, roles, people)
                out[i] = dm if dm is not None else self._fallback(o, spec, emails, best_person, roles)
            except Exception as e:
                log.warning("decision-maker step failed for %s: %s", o.name, e)
                o.flags.append("enrich_error")
        return out


def _country_names() -> Dict[str, str]:
    from brain.textutil import COUNTRY_CODES

    return {k: v for k, v in COUNTRY_CODES.items() if len(k) > 3}


_ATS = ("greenhouse.io", "lever.co", "ashbyhq.com", "workable.com", "smartrecruiters.com", "keka.com", "darwinbox",
        "zohorecruit", "freshteam.com", "recruitee.com", "breezy.hr", "wellfound.com/company")
_CAREERS = re.compile(r"\b(careers?|jobs|we'?re hiring|join (us|our team)|open (roles|positions))\b", re.I)


def hiring_signal(page: ParsedPage) -> float:
    """1.0 if the site links to a job board (open roles exist), 0.5 if it has a careers page, else 0."""
    links = page.nav_links + page.links
    if any(any(a in href.lower() for a in _ATS) for href, _ in links):
        return 1.0
    if any(_CAREERS.search(text or "") or re.search(r"/(careers?|jobs)(/|$)", href.lower()) for href, text in links):
        return 0.5
    return 0.0


_SERP_SPLIT = re.compile(r"\s+[-–—|·]\s+|\s+at\s+|,\s+")


SOCIAL_NOT_PROFILE = ("instagram.com", "facebook.com", "fb.com", "twitter.com", "x.com", "youtube.com", "tiktok.com",
                      "pinterest.com", "reddit.com", "quora.com", "medium.com")  # posts and reels, not a record of who works where


def _is_profile_url(url: str) -> bool:
    host = host_of(url or "")
    return not any(host == d or host.endswith("." + d) for d in SOCIAL_NOT_PROFILE)


def people_from_serp_scored(hits, org_name: str, roles: List[str]) -> Optional[Tuple[str, str, str, float]]:
    """(name, title, url, confidence) from search-result titles like "Priya Nair - Head of Partnerships - DevToolCo".
    Only result titles/snippets are read (no profile pages are fetched). Strict on purpose, because the name ends up in
    the greeting: the organisation must be named, a social-media post or reel is never a source, and a name with a
    repeated word ("Javier Javier") is rejected. Confidence is 0.9 when the whole organisation name appears together
    with a role in the title, 0.6 when only some of its (generic) words do, so "Photovoltaic Solar" cannot be vouched
    for by a stranger who is a "Photovoltaic Engineer"."""
    org_tokens = [t for t in re.findall(r"[a-z0-9]+", org_name.lower()) if len(t) > 2]
    org_phrase = " ".join(re.findall(r"[a-z0-9]+", org_name.lower()))
    for h in hits:
        if not _is_profile_url(h.url):
            continue
        title = (h.title or "").replace("| LinkedIn", "").replace("LinkedIn", "")
        blob = (title + " " + (h.snippet or "")).lower()
        if org_tokens and not any(t in blob for t in org_tokens):
            continue
        parts = [p.strip() for p in _SERP_SPLIT.split(title) if p.strip()]
        if len(parts) < 2:
            continue
        name = clean_name(parts[0])
        if not looks_like_person_name(name) or any(t in name.lower() for t in org_tokens):
            continue
        words = [w.lower().strip(".") for w in name.split()]
        if len(set(words)) < len(words):
            continue
        role_part = next((p for p in parts[1:] if _role_fits(p, roles)), None)
        role = role_part or next((r for r in roles if r.lower() in blob), None)
        if not role:
            continue
        phrase_in_title = bool(org_phrase) and org_phrase in " ".join(re.findall(r"[a-z0-9]+", title.lower()))
        conf = 0.9 if (role_part and phrase_in_title) else 0.6
        return name, squash(role, 80), h.url, conf
    return None


def people_from_serp(hits, org_name: str, roles: List[str]) -> Optional[Tuple[str, str, str]]:
    """A (name, title, url) from search-result titles (see people_from_serp_scored)."""
    found = people_from_serp_scored(hits, org_name, roles)
    return found[:3] if found else None
