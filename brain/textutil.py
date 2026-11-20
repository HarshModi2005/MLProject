"""Name, domain and country helpers shared across the brain. No network, no LLM."""

from __future__ import annotations

import re
import unicodedata
from typing import Tuple, List, Optional
from urllib.parse import urlsplit

HONORIFICS = {
    "dr", "prof", "professor", "mr", "mrs", "ms", "miss", "sir", "madam", "er", "shri", "smt",
    "associate", "assistant", "asst", "assoc", "phd", "ph.d", "md", "jr", "sr", "ii", "iii",
}

# Words that appear in navigation, headings and organisation names but never in a person's name.
NOT_NAME_WORDS = {
    # more section and corporate-page words ("Corporate Governance", "Frontline Workplace")
    "corporate", "governance", "frontline", "workplace", "workforce", "relations", "sustainability", "responsibility",
    "csr", "esg", "culture", "strategy", "operations", "transformation", "digital", "global", "worldwide", "business",
    "industries", "industrial", "financial", "investor", "stakeholders", "shareholders", "communication",
    "communications", "marketing", "sales", "advisory", "consulting", "learning", "academy", "foundation", "trust",
    "at", "by", "as", "is", "with", "from", "via", "about", "new", "latest",
    # page sections and company words that make page titles look like names ("Partner Spotlight")
    "partner", "partners", "partnership", "spotlight", "customer", "customers", "stories", "story", "sponsor",
    "sponsors", "member", "members", "community", "featured", "insights", "resources", "webinar", "webinars",
    "testimonials", "clients", "client", "portfolio", "investors", "mission", "vision", "values", "impact",
    "report", "updates", "newsroom", "gallery", "ecosystem", "support", "pricing", "features", "platform",
    "developers", "enterprise", "startups", "ventures", "capital", "holdings", "technologies", "studio",
    "solutions", "official", "limited", "pvt", "ltd", "inc", "llc",
    "department", "university", "institute", "college", "school", "faculty", "research", "home",
    "contact", "about", "people", "staff", "team", "news", "events", "publications", "projects",
    "teaching", "students", "alumni", "group", "lab", "laboratory", "centre", "center", "office",
    "admissions", "academics", "programs", "courses", "overview", "profile", "highlights",
    "directory", "menu", "search", "login", "careers", "jobs", "blog", "press", "privacy", "policy",
    "terms", "services", "products", "solutions", "company", "our", "the", "and", "of", "for",
    "read", "more", "view", "all", "click", "here", "email", "phone", "address", "science",
    "engineering", "computer", "technology", "management", "studies", "education", "library",
    "committee", "board", "meet", "leadership", "welcome", "page", "site", "map", "us", "in",
    "learning", "machine", "data", "systems", "intelligence", "artificial", "national", "indian",
    "international", "journal", "conference", "award", "awards", "google", "scholar", "linkedin",
    "twitter", "facebook", "github", "youtube", "instagram", "curriculum", "vitae", "biography",
    "interests", "areas", "education", "experience", "qualifications", "information", "details",
    "quick", "links", "main", "content", "skip", "to", "top", "back", "next", "previous",
    "academic", "calendar", "schedule", "notice", "notices", "circular", "tender", "brochure", "prospectus",
    "fee", "fees", "structure", "syllabus", "scheme", "regulations", "rules", "handbook", "report", "annual",
    "download", "downloads", "gallery", "photo", "photos", "video", "videos", "media", "resources", "forms",
    "apply", "application", "registration", "portal", "online", "campus", "hostel", "placement", "placements",
    "recruitment", "vacancy", "vacancies", "positions", "openings", "admission", "result", "results",
    "timetable", "examination", "exam", "exams", "seminar", "seminars", "workshop", "workshops", "lecture",
    "series", "colloquium", "talks", "visitors", "visiting", "emeritus", "adjunct", "chair", "dean", "head",
    "hod", "registrar", "director", "officer", "coordinator", "secretary", "cell", "club", "society",
    "association", "council", "senate", "division", "unit", "facility", "facilities", "infrastructure",
    "mission", "vision", "history", "overview", "message", "speech", "statement", "accreditation", "ranking",
    "rankings", "partners", "sponsors", "funding", "grants", "innovation", "incubation", "startup", "startups",
    "industry", "outreach", "community", "sitemap", "feedback", "disclaimer", "copyright", "accessibility",
    "dental", "dentistry", "clinic", "clinics", "care", "speciality", "specialty", "hospital", "hospitals",
    "health", "healthcare", "physio", "physiotherapy", "wellness", "medical", "diagnostics", "pharma",
    "pvt", "ltd", "llp", "inc", "llc", "private", "limited", "corp", "corporation", "group", "holdings",
    "ventures", "capital", "partners", "fund", "labs", "studio", "studios", "technologies", "tech", "software",
    "systems", "solutions", "digital", "agency", "consulting", "consultants", "enterprises", "industries",
    "foundation", "trust", "academy", "institute", "school", "bank", "finance", "insurance", "realty",
}

FREE_MAIL = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "protonmail.com", "icloud.com",
             "yahoo.co.in", "rediffmail.com", "live.com", "aol.com", "proton.me", "me.com"}

COUNTRY_CODES = {
    "india": "IN", "usa": "US", "us": "US", "united states": "US", "united states of america": "US",
    "america": "US", "uk": "GB", "united kingdom": "GB", "england": "GB", "britain": "GB",
    "great britain": "GB", "scotland": "GB", "germany": "DE", "canada": "CA", "australia": "AU",
    "singapore": "SG", "switzerland": "CH", "france": "FR", "netherlands": "NL", "japan": "JP",
    "china": "CN", "south korea": "KR", "korea": "KR", "italy": "IT", "spain": "ES", "sweden": "SE",
    "denmark": "DK", "norway": "NO", "finland": "FI", "ireland": "IE", "israel": "IL",
    "austria": "AT", "belgium": "BE", "new zealand": "NZ", "uae": "AE",
    "united arab emirates": "AE", "hong kong": "HK", "taiwan": "TW", "brazil": "BR",
    "portugal": "PT", "poland": "PL", "saudi arabia": "SA", "qatar": "QA", "south africa": "ZA",
    "mexico": "MX", "turkey": "TR", "greece": "GR", "czech republic": "CZ", "malaysia": "MY",
}

# Country-code TLDs that are routinely used as generic TLDs and say nothing about location.
_GENERIC_CC = {"io", "ai", "co", "me", "tv", "ly", "to", "fm", "gg", "cc", "sh", "so", "app", "dev"}
_SECOND_LEVEL = {"ac", "co", "com", "edu", "gov", "org", "net", "res", "nic", "ernet", "or", "ne", "go"}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def clean_name(raw: str) -> str:
    """'Prof. (Dr.) Sanjay K. Kumar, PhD' -> 'Sanjay K. Kumar'"""
    s = re.sub(r"\([^)]*\)", " ", raw or "")
    s = s.split(",")[0]
    tokens = [t for t in re.split(r"\s+", s.strip()) if t]
    while tokens and tokens[0].lower().strip(".") in HONORIFICS:
        tokens.pop(0)
    while tokens and tokens[-1].lower().strip(".") in HONORIFICS:
        tokens.pop()
    return " ".join(tokens)


def name_tokens(name: str) -> List[str]:
    s = strip_accents(clean_name(name)).lower()
    return [t for t in re.split(r"[^a-z]+", s) if t]


def looks_like_person_name(raw: str) -> bool:
    if not raw or len(raw) > 60 or any(ch.isdigit() for ch in raw) or "@" in raw:
        return False
    name = clean_name(raw)
    tokens = name.split()
    if not 2 <= len(tokens) <= 5:
        return False
    for t in tokens:
        bare = t.strip(".")
        if not bare or not re.fullmatch(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", bare):
            return False
        if bare.lower() in NOT_NAME_WORDS:
            return False
        if not (bare[0].isupper() or bare.lower() in {"de", "van", "von", "der", "bin", "al", "la", "di", "da"}):
            return False
    # an acronym next to ordinary words is a title or an organisation ("At SIM", "Program Officer ISB"), not a person;
    # a name written wholly in capitals ("RAJ KUMAR") is fine
    caps = [t for t in tokens if len(t.strip(".")) >= 3 and t.strip(".").isupper()]
    if caps and len(caps) < len(tokens):
        return False
    # at least one token longer than an initial
    return any(len(t.strip(".")) > 1 for t in tokens)


def host_of(url: str) -> str:
    return urlsplit(url if "//" in url else "//" + url).netloc.lower().split(":")[0].lstrip(".")


def registrable_domain(host_or_url: str) -> str:
    """Best-effort eTLD+1 without the public-suffix list: cs.iitrpr.ac.in -> iitrpr.ac.in"""
    host = host_of(host_or_url) if ("/" in host_or_url or ":" in host_or_url) else host_or_url.lower()
    host = host[4:] if host.startswith("www.") else host
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def country_from_domain(domain: str) -> Optional[str]:
    tld = domain.rsplit(".", 1)[-1].lower() if domain else ""
    if len(tld) != 2 or tld in _GENERIC_CC:
        return None
    return "GB" if tld == "uk" else tld.upper()


def to_country_code(value: str) -> Optional[str]:
    v = (value or "").strip()
    if len(v) == 2 and v.isalpha():
        return "GB" if v.upper() == "UK" else v.upper()
    return COUNTRY_CODES.get(v.lower())


def split_locations(values: List[str]):
    """['Punjab, India', 'USA'] -> (['IN', 'US'], ['Punjab'])"""
    countries: List[str] = []
    regions: List[str] = []
    for value in values:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if any(p.lower() in ("worldwide", "any", "anywhere", "global") for p in parts):
            continue
        for p in parts:
            code = to_country_code(p)
            if code:
                if code not in countries:
                    countries.append(code)
            elif p not in regions:
                regions.append(p)
    return countries, regions


def squash(text: str, limit: int = 0) -> str:
    s = re.sub(r"\s+", " ", text or "").strip()
    return s[:limit].rstrip() if limit and len(s) > limit else s


_REGION_ALIASES = [
    {"bangalore", "bengaluru"}, {"bombay", "mumbai"}, {"gurgaon", "gurugram"}, {"calcutta", "kolkata"},
    {"madras", "chennai"}, {"delhi", "new delhi"}, {"pune", "poona"}, {"mysore", "mysuru"},
    {"trivandrum", "thiruvananthapuram"}, {"cochin", "kochi"}, {"baroda", "vadodara"}, {"vizag", "visakhapatnam"},
    {"mohali", "sas nagar"}, {"noida", "gautam buddh nagar"},
]


# Metro areas named as one place: any member city counts as inside it.
_METROS = {
    "delhi ncr": ["delhi", "new delhi", "gurgaon", "gurugram", "noida", "ghaziabad", "faridabad", "greater noida"],
    "ncr": ["delhi", "new delhi", "gurgaon", "gurugram", "noida", "ghaziabad", "faridabad", "greater noida"],
    "mumbai metropolitan region": ["mumbai", "bombay", "thane", "navi mumbai", "kalyan"],
    "mmr": ["mumbai", "bombay", "thane", "navi mumbai", "kalyan"],
    "bay area": ["san francisco", "oakland", "berkeley", "palo alto", "mountain view", "sunnyvale", "san jose",
                 "menlo park", "redwood city", "santa clara", "cupertino"],
    "sf bay area": ["san francisco", "oakland", "berkeley", "palo alto", "mountain view", "sunnyvale", "san jose",
                    "menlo park", "redwood city", "santa clara", "cupertino"],
    "greater london": ["london"], "nyc": ["new york", "brooklyn", "manhattan"], "new york city": ["new york", "brooklyn", "manhattan"],
}
_ABBREVIATIONS = {"blr": "Bangalore", "bengaluru": "Bangalore", "hyd": "Hyderabad", "mum": "Mumbai", "bom": "Mumbai",
                  "sf": "San Francisco", "nyc": "New York City", "la": "Los Angeles", "dxb": "Dubai"}


def region_terms(region: str) -> List[str]:
    """'Bangalore' -> ['bangalore', 'bengaluru']; 'Delhi NCR' -> its member cities."""
    r = region.strip().lower()
    if r in _METROS:
        return sorted(set(_METROS[r] + [r]))
    for group in _REGION_ALIASES:
        if r in group:
            return sorted(group)
    return [r]


def normalise_locations(values: List[str]) -> Tuple[List[str], Optional[bool]]:
    """Clean what the user or the model wrote: 'near Indiranagar, BLR' -> 'Indiranagar, Bangalore'; 'remote' is not
    a place (returned as remote_ok=True). Returns (locations, remote_ok or None)."""
    out: List[str] = []
    remote: Optional[bool] = None
    for v in values or []:
        text = re.sub(r"^\s*(near|around|in|within \d+\s*km of)\s+", "", str(v), flags=re.I).strip()
        if not text:
            continue
        if re.fullmatch(r"(fully\s+)?remote( only| ok| friendly)?|anywhere|worldwide|global", text, re.I):
            remote = True
            continue
        parts = []
        for p in [x.strip() for x in text.split(",") if x.strip()]:
            if re.fullmatch(r"remote", p, re.I):
                remote = True
                continue
            parts.append(_ABBREVIATIONS.get(p.lower(), p))
        if parts:
            norm = ", ".join(parts)
            if norm.lower() not in {o.lower() for o in out}:
                out.append(norm)
    return out, remote


# Sites that list or write about organisations. They are sources, never targets, for hiring, partnership, sponsorship,
# fundraising or press asks ("Topstartups", "Clutch", "Crunchbase").
AGGREGATOR_DOMAINS = {
    "topstartups.io", "clutch.co", "goodfirms.co", "g2.com", "capterra.com", "crunchbase.com", "producthunt.com",
    "angel.co", "wellfound.com", "glassdoor.com", "ambitionbox.com", "tracxn.com", "growjo.com", "zoominfo.com",
    "apollo.io", "owler.com", "dnb.com", "yellowpages.com", "justdial.com", "sulekha.com", "indiamart.com",
    "tradeindia.com", "linkedin.com", "facebook.com", "instagram.com", "wikipedia.org", "medium.com", "quora.com",
    "reddit.com", "yourstory.com", "inc42.com", "startupindia.gov.in", "f6s.com", "dealroom.co", "pitchbook.com",
    "cbinsights.com", "betalist.com", "indeed.com", "naukri.com", "ycombinator.com", "themanifest.com",
}
_AGGREGATOR_NAME = re.compile(r"\b(directory|listings?|marketplace|top \d+|best \d+|list of|rankings?|reviews? of)\b|"
                              r"\btop[\s-]?startups\b", re.I)


def is_aggregator(name: str = "", domain: str = "") -> bool:
    """A directory, ranking or marketplace rather than an organisation that does the thing itself."""
    dom = registrable_domain(domain) if domain else ""
    if dom and dom in AGGREGATOR_DOMAINS:
        return True
    return bool(name and _AGGREGATOR_NAME.search(name))
