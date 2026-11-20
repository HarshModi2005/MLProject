"""
Organisation type: a free, heuristic classification of what a candidate is (an organisation) or where they
work (a person), and a gate that keeps types the campaign does not want out of the expensive stages.

    academic    universities, institutes, colleges
    media       news outlets, magazines, broadcasters
    publisher   academic / scientific publishers and journals
    directory   listings and aggregators (never a target)
    government  ministries, municipal bodies, public agencies
    nonprofit   NGOs, foundations, trusts
    company     everything else (always allowed)

A type is wanted when the playbook allows it (Playbook.allowed_org_classes) or the campaign's persona
asks for it in words ("professors", "journalists", "foundations"...). Unwanted types are demoted (their
pre-score cut, never enriched before wanted ones); directories are dropped. v2 shortlists showed why:
a university in an AI-startup job search, a publisher's staff in a press campaign.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from brain.models import CampaignSpec, Contact
from brain.textutil import registrable_domain

CLASSES = ("academic", "media", "publisher", "directory", "government", "nonprofit", "company")
ALWAYS = ("company", "unknown")

_ACADEMIC_DOMAIN = re.compile(r"(\.edu|\.ac\.[a-z]{2}|\.edu\.[a-z]{2})$")
_ACADEMIC_NAME = re.compile(r"\b(university|universit[äéà]|institute of (technology|science)|college|iit|iisc|iiit|nit|"
                            r"iiser|polytechnic|school of (engineering|management)|faculty of)\b", re.I)
_GOV_DOMAIN = re.compile(r"(\.gov|\.gov\.[a-z]{2}|\.nic\.in|\.gouv\.[a-z]{2}|\.govt\.[a-z]{2})$")
_GOV_NAME = re.compile(r"\b(ministry of|government of|municipal (corporation|council)|department of [a-z ]+, government)\b", re.I)
_PUBLISHERS = ("frontiersin.org", "springer.com", "elsevier.com", "sciencedirect.com", "mdpi.com", "wiley.com",
               "nature.com", "ieee.org", "acm.org", "tandfonline.com", "sagepub.com", "plos.org", "hindawi.com",
               "cambridge.org", "oup.com", "researchgate.net", "arxiv.org", "ssrn.com", "jstor.org")
_PUBLISHER_TEXT = re.compile(r"\b(peer[- ]reviewed|open[- ]access (journal|publisher)|scholarly publish|"
                             r"academic publish)\w*", re.I)
_MEDIA_NAME = re.compile(r"\b(news|times|magazine|herald|tribune|gazette|chronicle|express|daily|post|journal|"
                         r"today|weekly|media|broadcast(ing)?|tv|radio)\b", re.I)
_MEDIA_TEXT = re.compile(r"\b(news (website|portal|outlet|organi[sz]ation)|newspaper|news ?room|digital media (company|"
                         r"platform)|media (house|company|outlet)|magazine)\b", re.I)
# Job boards and startup lists named for what they list (topstartups.io, aijobs.net, startupslist.in ...).
_LISTING_DOMAIN = re.compile(r"(^|[-_])?(jobs?|careers|hiring|directory|listings?|topstartups|startupslist|"
                             r"startups-?list|companylist|toplist)([-_]|$)|^(top|best|list)[a-z]*(startups|companies|jobs)", re.I)
_NONPROFIT = re.compile(r"\b(non-?profit|ngo|not-for-profit|charitable (trust|society)|charity|foundation)\b", re.I)

# Words in the persona that ask for a type.
WANTS = {
    "academic": re.compile(r"\b(professors?|faculty|researchers?|academics?|universit(y|ies)|labs?|phd|postdoc|"
                           r"scientists?|colleges?|placement cells?)\b", re.I),
    "media": re.compile(r"\b(journalists?|reporters?|editors?|media|press|newsrooms?|publications?|podcasts?|"
                        r"writers?|bloggers?|newsletters?|creators?|influencers?)\b", re.I),
    "publisher": re.compile(r"\b(publishers?|journals?)\b", re.I),
    "government": re.compile(r"\b(government|ministr(y|ies)|public sector|municipal|officials?|tenders?|agenc(y|ies))\b", re.I),
    "nonprofit": re.compile(r"\b(ngos?|non-?profits?|foundations?|charit(y|ies)|trusts?|csr|philanthrop\w*|donors?)\b", re.I),
}


def classify(c: Contact, aggregator=None) -> Tuple[str, str]:
    """(class, why). People are classified by their organisation."""
    if aggregator is None:
        from brain.orgs import is_aggregator as aggregator
    domain = registrable_domain(c.org_domain) if c.org_domain else ""
    if not domain and c.best_email:
        domain = c.best_email.address.split("@")[-1]
    name = c.org or (c.name if c.kind == "org" else "") or ""
    text = " ".join([c.bio or ""] + [e.text for e in c.evidence[:2]])[:1500]
    if domain and aggregator(domain):
        return "directory", f"{domain} is a listing site"
    if domain and _LISTING_DOMAIN.search(domain.split(".")[0]):
        return "directory", f"{domain} looks like a job board or listing"
    if domain and any(domain.endswith(p) for p in _PUBLISHERS):
        return "publisher", f"{domain} is a publisher"
    if domain and _GOV_DOMAIN.search(domain):
        return "government", f"{domain} is a government domain"
    if domain and _ACADEMIC_DOMAIN.search(domain):
        return "academic", f"{domain} is an academic domain"
    if _ACADEMIC_NAME.search(name):
        return "academic", f"'{name}' is an academic institution"
    if _GOV_NAME.search(name):
        return "government", f"'{name}' is a government body"
    if domain:
        from brain.triage import NEWS_DOMAINS

        if any(n in domain for n in NEWS_DOMAINS):
            return "media", f"{domain} is a news outlet"
    if c.kind == "org" and _PUBLISHER_TEXT.search(text):
        return "publisher", "describes itself as a publisher"
    if (c.kind == "org" or c.org) and _MEDIA_NAME.search(name) and _MEDIA_TEXT.search(text):
        return "media", f"'{name}' describes itself as a media outlet"
    if c.kind == "org" and (_NONPROFIT.search(name) or re.search(r"\b(we are a|is a) (registered )?(non-?profit|ngo|charit)", text, re.I)):
        return "nonprofit", f"'{name}' is a non-profit"
    return "company", ""


def wanted_classes(spec: CampaignSpec, playbook=None) -> List[str]:
    out = list(ALWAYS) + list(getattr(playbook, "allowed_org_classes", None) or [])
    ask = " ".join([spec.persona.description or ""] + list(spec.persona.org_types) + list(spec.persona.roles))
    out += [k for k, rx in WANTS.items() if rx.search(ask)]
    return list(dict.fromkeys(out))


OFF_TYPE = "off_type:"


def gate(contacts: List[Contact], spec: CampaignSpec, playbook=None, aggregator=None
         ) -> Tuple[List[Contact], Dict[str, int]]:
    """Tag every candidate with its class; flag unwanted ones (rank.prescore demotes flagged candidates) and drop
    directories. Returns (kept, counts per class that was demoted or dropped)."""
    wanted = set(wanted_classes(spec, playbook))
    kept, counts = [], {}
    for c in contacts:
        cls, why = classify(c, aggregator)
        c.flags = [f for f in c.flags if not f.startswith(("org_class:", OFF_TYPE))] + [f"org_class:{cls}"]
        if cls == "directory" and "directory" not in wanted:
            counts["dropped_directory"] = counts.get("dropped_directory", 0) + 1
            continue
        if cls not in wanted:
            c.flags.append(OFF_TYPE + cls)
            counts[f"demoted_{cls}"] = counts.get(f"demoted_{cls}", 0) + 1
        kept.append(c)
    return kept, counts


def off_type(c: Contact) -> Optional[str]:
    return next((f[len(OFF_TYPE):] for f in c.flags if f.startswith(OFF_TYPE)), None)
