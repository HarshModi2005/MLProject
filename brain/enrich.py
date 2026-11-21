"""
Enrichment: find a trustworthy email (and richer context) for candidates that already ranked well.

Order of attack, cheapest first, stopping as soon as a confident address is found:
  1. ORCID public record            (free API)
  2. the person's own known pages    (page fetch)
  3. one targeted web search         (search API call) -> fetch the best 1-2 results
Then, across the whole pool: learn each organisation's address pattern from confirmed emails and
use it to propose (clearly flagged, low-confidence) addresses for people still missing one.
No LLM calls.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from brain.emailfinder import (
    apply_patterns, attach_emails, infer_domain_patterns, name_match, verify_deliverable,
)
from brain.fetch import Fetcher
from brain.models import Contact, Evidence
from brain.pages import INTEREST_RE, classify, page_person_name, parse, _first_title
from brain.resolve import names_compatible
from brain.settings import Settings
from brain.sitecrawl import crawl_org
from brain.sources.orcid import Orcid
from brain.sources.websearch import WebSearch
from brain.textutil import country_from_domain, name_tokens, registrable_domain, squash

log = logging.getLogger("brain.enrich")
_NOT_PERSONAL = ("openalex.org", "orcid.org", "doi.org")
# Platforms that host many people's profiles: a page there is not "their own site" for the trusted reading above.
_PLATFORMS = ("github.com", "gitlab.com", "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
              "medium.com", "npmjs.com", "pypi.org", "huggingface.co", "youtube.com", "researchgate.net",
              "scholar.google", "semanticscholar.org", "wikipedia.org", "dev.to", "substack.com")
_CONTACT_LINK = re.compile(r"/(contact|about|me|hire|work-with-me)(/|\.html?)?$|\b(contact|about me|about|hire me|get in touch)\b", re.I)


def has_usable_email(c: Contact, threshold: float) -> bool:
    from brain.discovery.reach.tiers import email_tier

    best = c.best_email
    # A confident-looking address with nothing tying it to this person (a namesake, a previous employer) is not reachable.
    return bool(best and best.confidence >= threshold and email_tier(c, best) not in ("unattributed",))


def absorb_page(contact: Contact, html: str, url: str, trusted: bool = False) -> bool:
    """Merge what a fetched page says into the contact. Returns True if the page is about them.
    `trusted`: the page is the person's own site, reached from a link their own profile gives (a GitHub blog field, a
    package homepage). A personal site rarely looks like a "profile page" to the classifier (a blog home, a one-page
    portfolio), so its addresses are taken unless it lists many people."""
    page = parse(html, url)
    name = page_person_name(page)
    own = bool(name and names_compatible(name, contact.name)) and classify(page) == "profile"
    if trusted and not own and 0 < len(page.emails) <= 6 and not (classify(page) == "directory"):
        attach_emails(contact, page.mailtos, "mailto", url, on_own_page=True)
        attach_emails(contact, [e for e in page.emails if e not in page.mailtos], "text", url, on_own_page=True)
        contact.add_url(url)
        return True
    if own and len(page.emails) > 6:
        own = False  # a page listing many addresses is not one person's page, whatever its title says
    if own:
        attach_emails(contact, page.mailtos, "mailto", url, on_own_page=True)
        attach_emails(contact, [e for e in page.emails if e not in page.mailtos], "text", url, on_own_page=True)
        contact.add_url(url)
        domain = registrable_domain(url)
        contact.org_domain = contact.org_domain or domain
        contact.country = contact.country or country_from_domain(domain)
        contact.title = contact.title or _first_title(page.text)
        if not contact.topics:
            m = INTEREST_RE.search(page.text)
            if m:
                contact.topics = [squash(t, 60) for t in m.group(1).replace(";", ",").split(",") if 2 < len(t.strip()) < 60][:10]
        paragraphs = [squash(p) for p in page.text.split("\n") if len(p.strip()) > 120]
        if paragraphs and not contact.bio:
            contact.bio = paragraphs[0][:600]
            contact.evidence.append(Evidence(kind="page", text=paragraphs[0][:400], url=url))
    else:
        # Someone else's page (e.g. a department listing): only take addresses that carry this person's name.
        mine = [e for e in page.emails if name_match(contact.name, e) == 1.0]
        if len(mine) <= 2:  # more than two "matches" on one page means the name is ambiguous there
            attach_emails(contact, mine, "text", url, on_own_page=False)
    return own


class Enricher:
    def __init__(self, fetcher: Fetcher, search: WebSearch, orcid: Orcid, settings: Optional[Settings] = None,
                 store=None, accept_prior: bool = False):
        self.store, self.accept_prior = store, accept_prior  # cross-campaign email conventions (reach/patterns.py)
        self.fetcher = fetcher
        self.search = search
        self.orcid = orcid
        self.settings = settings or Settings()
        self.stats: Dict[str, int] = {"orcid_lookups": 0, "pages": 0, "searches": 0, "email_from_orcid": 0,
                                      "email_from_page": 0, "email_from_search": 0, "email_guessed": 0,
                                      "undeliverable_removed": 0}

    def _enrich_one(self, c: Contact) -> None:
        threshold = self.settings.min_email_confidence
        if has_usable_email(c, threshold):
            return
        # 1. ORCID
        orcid = next((k.split(":", 1)[1] for k in c.keys if k.startswith("orcid:")), None)
        if orcid:
            self.stats["orcid_lookups"] += 1
            emails, urls = self.orcid.person(orcid)
            if attach_emails(c, emails, "orcid", f"https://orcid.org/{orcid}", on_own_page=True):  # self-declared
                self.stats["email_from_orcid"] += 1
            for u in urls[:3]:
                c.add_url(u)
        # 2. Known pages
        if not has_usable_email(c, threshold):
            for url in [u for u in c.urls if not any(h in u for h in _NOT_PERSONAL)][:3]:
                own_site = not any(h in url.lower() for h in _PLATFORMS)
                page = self.fetcher.get(url)
                if page.ok:
                    self.stats["pages"] += 1
                    absorb_page(c, page.html, page.url, trusted=own_site)
                    if own_site and not has_usable_email(c, threshold):
                        self._follow_contact_pages(c, page, threshold)
                if has_usable_email(c, threshold):
                    self.stats["email_from_page"] += 1
                    break
        # 3. Targeted search
        if not has_usable_email(c, threshold) and self.search.available:
            self.stats["searches"] += 1
            query = f'"{c.name}" {c.org or ""} email'.strip()
            surname = (name_tokens(c.name) or [""])[-1]
            hits = self.search.search(query, n=6)
            if self._emails_from_snippets(c, hits) and has_usable_email(c, threshold):
                self.stats["email_from_snippet"] = self.stats.get("email_from_snippet", 0) + 1
                return

            def priority(h) -> int:
                same_org = bool(c.org_domain) and registrable_domain(h.url) == registrable_domain(c.org_domain)
                names_them = surname in h.title.lower() or surname in h.url.lower()
                return (2 if same_org else 0) + (1 if names_them else 0)

            for h in sorted(hits, key=priority, reverse=True)[:2]:
                if priority(h) == 0:
                    break
                page = self.fetcher.get(h.url)
                if page.ok:
                    self.stats["pages"] += 1
                    absorb_page(c, page.html, page.url)
                if has_usable_email(c, threshold):
                    self.stats["email_from_search"] += 1
                    break

    def _follow_contact_pages(self, c: Contact, page, threshold: float, limit: int = 2) -> None:
        """A personal site often keeps its address on /contact or /about, one link from the home page."""
        parsed = parse(page.html, page.url)
        host = registrable_domain(page.url)
        seen = {page.url.rstrip("/")}
        for href, text in parsed.nav_links:
            if len(seen) > limit:
                break
            if registrable_domain(href) != host or href.rstrip("/") in seen:
                continue
            if not _CONTACT_LINK.search(href) and not _CONTACT_LINK.search(text or ""):
                continue
            seen.add(href.rstrip("/"))
            sub = self.fetcher.get(href)
            if sub.ok:
                self.stats["pages"] += 1
                absorb_page(c, sub.html, sub.url, trusted=True)
            if has_usable_email(c, threshold):
                return

    def _emails_from_snippets(self, c: Contact, hits) -> int:
        """Search-result snippets often quote an address ("... email: jane@lab.org"), which saves fetching a page that
        will not load. Taken only when the address carries the person's name or is on their organisation's domain."""
        from brain.emailfinder import extract_emails

        added = 0
        for h in hits[:6]:
            for addr in extract_emails(f"{h.title} {h.snippet}"):
                same_org = bool(c.org_domain) and registrable_domain(addr.split("@")[1]) == registrable_domain(c.org_domain)
                if name_match(c.name, addr) >= 1.0 or (same_org and name_match(c.name, addr) > 0):
                    added += attach_emails(c, [addr], "text", h.url, on_own_page=False)
        return added

    def enrich(self, targets: List[Contact], pool: Optional[List[Contact]] = None,
               hint_terms: Optional[List[str]] = None) -> None:
        """Enrich `targets`. `pool` (default: targets) is the wider set used to learn address patterns."""
        with ThreadPoolExecutor(max_workers=max(1, self.settings.fetch_workers)) as ex:
            for c, err in zip(targets, ex.map(self._safe, targets)):
                if err:
                    log.warning("enrichment failed for %s: %s", c.name, err)
                    c.flags.append("enrich_error")
        self._crawl_org_sites(targets, hint_terms or [])
        patterns = infer_domain_patterns(pool or targets)
        self.stats["email_guessed"] = apply_patterns(targets, patterns)
        from brain.discovery.reach.patterns import fill

        self.stats["email_guessed_known_convention"] = fill(targets, self.store, self.accept_prior)
        self.stats["undeliverable_removed"] = verify_deliverable(targets)

    def _crawl_org_sites(self, targets: List[Contact], hint_terms: List[str], max_orgs: int = 8) -> None:
        """For organisations whose candidates still lack an email, read the staff directories on the
        organisation's own site and match the people listed there back to those candidates."""
        threshold = self.settings.min_email_confidence
        missing: Dict[str, List[Contact]] = {}
        for c in targets:
            if c.org_domain and not has_usable_email(c, threshold):
                missing.setdefault(registrable_domain(c.org_domain), []).append(c)
        # Organisations with the most unreachable candidates first: best return per page fetched.
        chosen = sorted(missing.items(), key=lambda kv: len(kv[1]), reverse=True)[:max_orgs]

        def crawl(domain: str) -> List[Contact]:
            try:
                return crawl_org(f"https://{domain}/", self.fetcher, hint_terms)
            except Exception as e:
                log.warning("site crawl failed for %s: %s", domain, e)
                return []

        # Different hosts, so crawling them side by side stays polite to each.
        with ThreadPoolExecutor(max_workers=max(1, min(len(chosen), self.settings.fetch_workers))) as ex:
            crawled = list(ex.map(crawl, [d for d, _ in chosen])) if chosen else []
        for (domain, people), listed in zip(chosen, crawled):
            self.stats["org_sites_crawled"] = self.stats.get("org_sites_crawled", 0) + 1
            for c in people:
                for other in listed:
                    if other.emails and names_compatible(other.name, c.name):
                        for e in other.emails:
                            c.add_email(e)
                        for u in other.urls:
                            c.add_url(u)
                        if has_usable_email(c, threshold):
                            self.stats["email_from_org_site"] = self.stats.get("email_from_org_site", 0) + 1
                        break

    def _safe(self, c: Contact) -> Optional[str]:
        try:
            self._enrich_one(c)
            return None
        except Exception as e:  # one bad page must not abort the batch
            return f"{type(e).__name__}: {e}"
