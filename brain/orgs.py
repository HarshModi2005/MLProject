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
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from brain.discover import rank_hits
from brain.emailfinder import extract_emails, is_role_address, name_match
from brain.fetch import Fetcher
from brain.llm import LLM, LLMError
from brain.models import CampaignSpec, Contact, EmailCandidate, Evidence
from brain.packs import Pack
from brain.pages import classify, directory_contacts, parse, profile_contact
from brain.settings import Settings
from brain.sources.websearch import Hit, WebSearch
from brain.textutil import (
    FREE_MAIL, country_from_domain, host_of, region_terms, registrable_domain, squash, to_country_code,
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
    user = f"TARGET: {target}\nPAGE: {url}\n\nTEXT:\n{page.text[:6000]}\n\nLINKS:\n{link_lines or '(none)'}"
    data = llm.json(system=ORG_LIST_SYSTEM, user=user, purpose="extract_orgs", tier="fast", max_tokens=2500)
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
                  max_list_pages: int = 8) -> Tuple[List[Contact], Dict[str, int]]:
    """Organisations from web search. Each query's results are processed at once (own sites kept, list pages
    read) so the planner knows what every query yielded and can stop as soon as there are enough."""
    from brain.context import CampaignContext
    from brain.searchplan import SearchPlanner

    stats = {"queries": 0, "hits": 0, "org_sites": 0, "list_pages": 0, "orgs_from_lists": 0}
    if not search.available:
        return [], stats
    planner = planner or SearchPlanner(spec, pack, CampaignContext(), llm)
    orgs: Dict[str, Contact] = {}
    seen_urls: set = set()

    def add(c: Contact) -> bool:
        key = c.org_domain or c.name.lower()
        if key in orgs:
            orgs[key].evidence += [e for e in c.evidence if e.text not in {x.text for x in orgs[key].evidence}]
            return False
        orgs[key] = c
        return True

    def execute(query: str) -> Tuple[int, int]:
        stats["queries"] += 1
        hits = [h for h in search.search(query, n=10) if h.url not in seen_urls]
        seen_urls.update(h.url for h in hits)
        stats["hits"] += len(hits)
        new = 0
        lists: List[Hit] = []
        for _, h in rank_hits(hits, spec, pack):
            dom = registrable_domain(h.url)
            if is_aggregator(dom) or _LIST_TITLE.search(h.title or ""):
                lists.append(h)
                continue
            home = f"{urlsplit(h.url).scheme or 'https'}://{host_of(h.url)}/"
            c = _org_contact(_org_name_from_title(h.title, dom), dom, home, h.snippet)
            c.add_url(h.url)
            if add(c):
                new += 1
                stats["org_sites"] += 1
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
                    if add(c):
                        new += 1
                        stats["orgs_from_lists"] += 1
        return len(hits), new

    planner.run(execute, have=lambda: len(orgs), budget_left=lambda: search.remaining - 2, kind="search")
    return list(orgs.values()), stats


def discover_places(spec: CampaignSpec, pack: Pack, search: WebSearch, planner: "SearchPlanner",
                    gl: str = "") -> Tuple[List[Contact], Dict[str, int]]:
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
            orgs[dom] = c
            new += 1
        stats["with_website"] = len(orgs)
        return len(found), new

    planner.run(execute, have=lambda: len(orgs), budget_left=lambda: search.remaining - 2, kind="places",
                first_round_max=6)
    return list(orgs.values()), stats


# ── enrichment ───────────────────────────────────────────────────────────────

def _inbox_rank(address: str, preferred: List[str]) -> Optional[int]:
    """Lower is better; None means this address must not be used."""
    tokens = [t for t in re.split(r"[._\-+]", address.split("@")[0].lower()) if t]
    if not tokens or any(t in NEVER_INBOXES for t in tokens) and not any(t in preferred for t in tokens):
        return None
    order = [p.lower() for p in preferred] + [d for d in DEFAULT_INBOXES if d not in preferred]
    ranks = [order.index(t) for t in tokens if t in order]
    return min(ranks) if ranks else None


def _role_fits(title: Optional[str], roles: List[str]) -> bool:
    if not title or not roles:
        return False
    t = title.lower()
    return any(r.lower() in t or t in r.lower() for r in roles)


class OrgEnricher:
    def __init__(self, fetcher: Fetcher, search: WebSearch, settings: Settings):
        self.fetcher, self.search, self.settings = fetcher, search, settings
        self.stats = {"orgs_read": 0, "sites_resolved": 0, "person_contacts": 0, "inbox_contacts": 0,
                      "site_contacts": 0, "no_contact": 0}

    def _resolve_site(self, org: Contact, spec: CampaignSpec) -> None:
        """Find the official website of an organisation known only by name (one search)."""
        place = (spec.regions or [""])[0]
        hits = self.search.search(f'"{org.name}" {place} official website', n=5)
        tokens = [t for t in re.findall(r"[a-z0-9]+", org.name.lower()) if len(t) > 2]
        for h in hits:
            dom = registrable_domain(h.url)
            label = dom.split(".")[0]
            if is_aggregator(dom):
                continue
            if any(t in label for t in tokens) or label in "".join(tokens):
                org.org_domain = dom
                org.country = org.country or country_from_domain(dom)
                org.add_url(f"https://{host_of(h.url)}/")
                self.stats["sites_resolved"] += 1
                return

    def enrich(self, org: Contact, spec: CampaignSpec, roles: List[str]) -> Contact:
        if not org.org_domain:
            self._resolve_site(org, spec)
        if not org.org_domain:
            org.flags.append("no_website")
            return org
        home = next((u for u in org.urls if registrable_domain(u) == org.org_domain and urlsplit(u).path in ("", "/")),
                    f"https://{org.org_domain}/")
        page = self.fetcher.get(home)
        if not page.ok and "//www." not in home:
            page = self.fetcher.get(home.replace("://", "://www.", 1))
        if not page.ok:
            org.flags.append("site_unreachable")
            return org
        self.stats["orgs_read"] += 1
        parsed = parse(page.html, page.url)
        pages = [parsed]
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
        if description:
            org.bio = description[:400]
            org.evidence.insert(0, Evidence(kind="profile", text=description[:300], url=page.url))

        # Follow the few internal pages most likely to name people or give contact details.
        scored: Dict[str, int] = {}
        for href, text in parsed.nav_links:
            if registrable_domain(href) != org.org_domain or _AVOID.search(href) or href.rstrip("/") == page.url.rstrip("/"):
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
        if best_person:
            person, good = best_person
            c = Contact(kind="person", name=person.name, title=person.title, org=org.name, org_domain=org.org_domain,
                        country=org.country, location_text=org.location_text, urls=person.urls + org.urls,
                        evidence=org.evidence + person.evidence, bio=org.bio, sources=org.sources, flags=org.flags)
            for e in good:
                c.add_email(e)
            self.stats["person_contacts"] += 1
            return c

        # 2. The organisation's shared inbox, chosen to suit the ask (partnerships@ for sponsorship, ...).
        preferred = [p.lower() for p in spec.preferred_inboxes]
        candidates = []
        for addr, src in emails.items():
            dom = registrable_domain(addr.split("@")[1])
            if dom != org.org_domain:
                continue
            rank = _inbox_rank(addr, preferred)
            if rank is not None and is_role_address(addr) or rank is not None and rank < len(preferred):
                candidates.append((rank, addr, src))
        if candidates:
            _, addr, src = min(candidates)
            org.add_email(EmailCandidate(address=addr, confidence=0.7, method="inbox", source_url=src))
            self.stats["inbox_contacts"] += 1
            return org

        # 3. Small organisations often publish one personal-looking address (the owner's): use it if it is the only one.
        personal = [a for a in emails if not is_role_address(a) and _inbox_rank(a, preferred) is None
                    and registrable_domain(a.split("@")[1]) in (org.org_domain, *FREE_MAIL)]
        if len(personal) == 1:
            org.add_email(EmailCandidate(address=personal[0], confidence=0.65, method="site", source_url=emails[personal[0]]))
            self.stats["site_contacts"] += 1
            return org
        self.stats["no_contact"] += 1
        return org

    def enrich_all(self, orgs: List[Contact], spec: CampaignSpec, pack: Pack) -> List[Contact]:
        roles = spec.persona.roles or pack.default_roles

        def one(o: Contact) -> Contact:
            try:
                return self.enrich(o, spec, roles)
            except Exception as e:  # one broken site must not abort the batch
                log.warning("org enrichment failed for %s: %s", o.name, e)
                o.flags.append("enrich_error")
                return o

        with ThreadPoolExecutor(max_workers=max(1, self.settings.fetch_workers)) as ex:
            return list(ex.map(one, orgs))


def _country_names() -> Dict[str, str]:
    from brain.textutil import COUNTRY_CODES

    return {k: v for k, v in COUNTRY_CODES.items() if len(k) > 3}
