"""
Email extraction and confidence scoring. No LLM.

An address is only useful if it plausibly belongs to *this person*, so every candidate is scored
against the contact's name and organisation domain. Role inboxes are rejected outright.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from brain.models import Contact, EmailCandidate
from brain.textutil import FREE_MAIL, name_tokens, registrable_domain

EMAIL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,12}")

ROLE_LOCALPARTS = {
    "info", "admin", "administrator", "office", "webmaster", "web", "contact", "contactus", "support",
    "hr", "careers", "jobs", "admissions", "admission", "registrar", "dean", "hod", "head", "enquiry",
    "enquiries", "inquiry", "help", "helpdesk", "sales", "noreply", "no-reply", "donotreply",
    "postmaster", "press", "media", "privacy", "abuse", "security", "team", "hello", "mail", "email",
    "marketing", "feedback", "library", "director", "principal", "chairman", "pro", "reception",
    "accounts", "billing", "legal", "newsletter", "events", "placement", "placements", "academics",
    "example", "name", "username", "user", "firstname.lastname", "your.name", "test",
}

_AT = r"(?:\s*[\[\(\{<]\s*(?:at|AT|At)\s*[\]\)\}>]\s*|\s+(?:AT|at)\s+|\s*\[\s*@\s*\]\s*|\s*\(\s*@\s*\)\s*)"
_DOT = r"(?:\s*[\[\(\{<]\s*(?:dot|DOT|Dot)\s*[\]\)\}>]\s*|\s+(?:DOT|dot)\s+)"
_OBFUSCATED = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._%+\-]*)" + _AT + r"([A-Za-z0-9\-]+(?:(?:\.|" + _DOT + r")[A-Za-z0-9\-]+)+)"
)


def deobfuscate(text: str) -> str:
    """'sk [at] iitrpr [dot] ac [dot] in' / 'sk at iitrpr dot ac dot in' -> 'sk@iitrpr.ac.in'"""

    def fix(m: "re.Match[str]") -> str:
        domain = re.sub(_DOT, ".", m.group(2))
        return f"{m.group(1)}@{domain}"

    return _OBFUSCATED.sub(fix, text or "")


def decode_cfemail(hexstr: str) -> str:
    """Cloudflare's email protection stores 'user@host' XOR-ed with the first byte as hex; '' if it is not valid."""
    try:
        raw = bytes.fromhex((hexstr or "").strip())
        if len(raw) < 2:
            return ""
        key = raw[0]
        return bytes(b ^ key for b in raw[1:]).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return ""


def cloudflare_emails(soup) -> List[str]:
    """Addresses hidden by Cloudflare email protection (data-cfemail / '/cdn-cgi/l/email-protection#hex' links)."""
    out: List[str] = []
    hexes = [el.get("data-cfemail", "") for el in soup.find_all(attrs={"data-cfemail": True})]
    hexes += [a["href"].split("#", 1)[1] for a in soup.find_all("a", href=True)
              if "email-protection#" in a["href"]]
    for h in hexes:
        for e in extract_emails(decode_cfemail(h)):
            if e not in out:
                out.append(e)
    return out


def extract_emails(text: str) -> List[str]:
    found = EMAIL_RE.findall(deobfuscate(text))
    out: List[str] = []
    for e in found:
        e = e.strip(".").lower()
        if e.rsplit(".", 1)[-1] in {"png", "jpg", "jpeg", "gif", "svg", "webp", "css", "js"}:
            continue
        if e not in out:
            out.append(e)
    return out


ROLE_TOKENS = ROLE_LOCALPARTS | {
    "academics", "relations", "phd", "office", "dept", "department", "registry", "exam", "exams", "hostel",
    "alumni", "scholarship", "research", "dir", "chair", "secy", "secretary", "convener", "coordinator",
    "tpo", "iro", "international", "outreach", "admissions", "faculty", "staff", "hr", "recruitment",
    "token", "api", "demo", "sample", "foo", "bar", "johndoe", "janedoe", "someone", "username",
}


def is_role_address(address: str) -> bool:
    """True for shared inboxes (office@, dean.academics@, phd-admissions@ ...)."""
    local = address.split("@")[0].lower()
    if local in ROLE_LOCALPARTS:
        return True
    tokens = [t for t in re.split(r"[._\-+]", local) if t]
    return any(t in ROLE_TOKENS for t in tokens)


def _explained(part: str, tokens: List[str], patterns) -> bool:
    """Is this piece of an address's local part accounted for by the person's name (or a pattern of it)?"""
    if part in tokens or any(part in pat for pat in patterns):
        return True
    return any(len(t) >= 4 and (t in part) for t in tokens)  # 'drrahul', 'sharmacs'


def name_match(name: str, address: str) -> float:
    """How strongly an address's local part matches a person's name: 0, 0.5 (initials) or 1.

    Matching is on whole parts of the local part (split at dots, digits, dashes), never loose substrings:
    'gaur' must not match 'gauravsingal789'."""
    tokens = name_tokens(name)
    if not tokens:
        return 0.0
    raw_local = address.split("@")[0].lower()
    parts = [p for p in re.split(r"[^a-z]+", raw_local) if p]
    joined = "".join(parts)
    if not parts:
        return 0.0
    first, last = tokens[0], tokens[-1]
    patterns = {first + last, last + first, first[0] + last, first + last[0], last + first[0]}
    if len(tokens) > 2:
        patterns |= {"".join(tokens), first + tokens[1] + last, "".join(t[0] for t in tokens[:-1]) + last}
    # Only first and last names identify someone: middle names such as Kumar, Singh or Kaur are shared by millions.
    full = {t for t in (first, last) if len(t) >= 3}
    # 'atul.kumar' is a different person called Kumar: a part that is some other name rules the match out.
    others = [p for p in parts if len(p) >= 3 and p not in tokens and not any(p in pat for pat in patterns)]
    if last in parts and first not in joined and others:
        return 0.0
    # A full-name pattern ('rahulsharma', 'sharmar', 'drtusharshukla') settles it before anything looser is tried.
    for part in parts + [joined]:
        if part in patterns or any(len(pat) >= 8 and pat in part for pat in patterns):
            return 1.0
    # The first name alone is only a match when nothing contradicts it: 'rahul@' can be Rahul Sharma, but
    # 'rahul.gupta@' and 'shampa.santra@' carry another surname, so they belong to someone else.
    if len(first) >= 3 and first in parts:
        return 0.0 if [p for p in parts if len(p) >= 3 and p != first and not _explained(p, tokens, patterns)] else 1.0
    for part in parts + [joined]:
        if part in patterns or (part == first and len(first) >= 3):
            return 1.0
        if part == last and len(last) >= 3:
            return 1.0 if len(parts) == 1 and len(last) >= 7 else 0.5  # 'kumar@' alone is ambiguous
        # prefixes/suffixes such as 'dr' + name, or name + an extra surname: 'drtusharshukla', 'manindersohal'
        if any(len(pat) >= 8 and pat in part for pat in patterns):
            return 1.0
        if any(len(t) >= 5 and (part.startswith(t) or part.endswith(t)) for t in full):
            return 1.0
    initials = "".join(t[0] for t in tokens)
    if joined in {initials, initials[::-1]} or (len(joined) <= 4 and joined[0] == first[0]
                                                 and (len(tokens) == 1 or last[0] in joined[1:])):
        return 0.5
    return 0.0


DIFFERENT_ORG = 0.45  # penalty: address at another organisation's domain, not from the person's own page
DIFFERENT_ORG_OWN_PAGE = 0.15  # same, but the person's own page lists it (a second affiliation is plausible)

_METHOD_BASE = {"provided": 0.95, "orcid": 0.9, "mailto": 0.8, "text": 0.7, "directory": 0.75, "llm": 0.65,
                "pattern": 0.4}


_ACADEMIC_SUFFIX = (".edu", ".ac.in", ".ac.uk", ".edu.in", ".edu.au", ".ac.jp", ".ac.kr", ".edu.cn", ".ac.nz", ".ac.za")


def is_institutional(domain: str) -> bool:
    return domain.endswith(_ACADEMIC_SUFFIX) or ".ac." in domain or ".edu." in domain


_GENERIC_ORG_WORDS = {"india", "indian", "research", "institute", "university", "college", "global", "group",
                      "technology", "technologies", "international", "national", "science", "sciences", "school"}


def _org_mentions(org: str, domain: str) -> bool:
    """Does the domain plausibly belong to the organisation name? 'Thapar Institute' ~ thapar.edu"""
    label = domain.split(".")[0]
    words = [w for w in re.findall(r"[a-z]+", org.lower()) if len(w) > 2]
    initials = "".join(w[0] for w in words if w not in {"of", "and", "the", "for"})
    distinctive = [w for w in words if len(w) > 3 and w not in _GENERIC_ORG_WORDS]  # 'india' names no organisation
    return any(w in label or label in w for w in distinctive) or label in (initials, initials[:len(label)])


def score_email(address: str, contact: Contact, method: str, on_own_page: bool = False) -> float:
    """Confidence in [0, 1] that `address` reaches `contact`. 0 means reject."""
    address = address.lower()
    if is_role_address(address):
        return 0.0
    if method in ("provided", "pattern"):
        return _METHOD_BASE[method]
    domain = address.split("@")[1]
    score = _METHOD_BASE.get(method, 0.6)
    nm = name_match(contact.name, address)
    if nm == 1.0:
        score += 0.15
    elif nm == 0.5:
        score += 0.05
    else:
        # An unrelated address on someone's own page is often still theirs (nicknames);
        # anywhere else it most likely belongs to a different person.
        score -= 0.15 if on_own_page else 0.45
    reg = registrable_domain(domain)
    if contact.org_domain and reg == registrable_domain(contact.org_domain):
        score += 0.1
    elif reg in FREE_MAIL:
        score -= 0.05 if nm == 1.0 else 0.2
    elif contact.org_domain:
        # A different organisation's domain: a previous employer or a namesake. Only the person's own page makes it
        # plausible; anywhere else it must not clear the send threshold on a name match alone (a "Rahul Sharma" at
        # Microsoft Research is not the Rahul Sharma at stanford.edu).
        score -= DIFFERENT_ORG_OWN_PAGE if on_own_page else DIFFERENT_ORG
    elif is_institutional(reg) and contact.org and not _org_mentions(contact.org, reg):
        score -= DIFFERENT_ORG_OWN_PAGE if on_own_page else DIFFERENT_ORG  # organisation known only by name
    return max(0.0, min(1.0, round(score, 2)))


def attach_emails(contact: Contact, addresses: Iterable[str], method: str, source_url: Optional[str],
                  on_own_page: bool = False) -> int:
    added = 0
    addresses = list(addresses)
    # If one address on the page clearly carries the person's name, the others belong to someone else.
    if on_own_page and any(name_match(contact.name, a) > 0 for a in addresses):
        on_own_page = False
    for addr in addresses:
        conf = score_email(addr, contact, method, on_own_page=on_own_page)
        if conf > 0:
            contact.add_email(EmailCandidate(address=addr, confidence=conf, method=method, source_url=source_url))
            added += 1
    return added


# ── pattern inference ────────────────────────────────────────────────────────

def _patterns(name: str) -> Dict[str, str]:
    t = name_tokens(name)
    if len(t) < 2:
        return {}
    f, l = t[0], t[-1]
    return {
        "first.last": f"{f}.{l}", "firstlast": f"{f}{l}", "flast": f"{f[0]}{l}", "first": f,
        "last": l, "first_last": f"{f}_{l}", "lastf": f"{l}{f[0]}", "firstl": f"{f}{l[0]}",
        "last.first": f"{l}.{f}", "f.last": f"{f[0]}.{l}",
    }


def infer_domain_patterns(contacts: Iterable[Contact], min_support: int = 3) -> Dict[str, Tuple[str, int]]:
    """Learn each organisation's address convention from contacts whose emails we already trust."""
    votes: Dict[str, Counter] = {}
    for c in contacts:
        best = c.best_email
        if not best or best.method == "pattern" or best.confidence < 0.8:
            continue
        local, domain = best.address.split("@")
        for pname, value in _patterns(c.name).items():
            if value == local:
                votes.setdefault(domain, Counter())[pname] += 1
    out: Dict[str, Tuple[str, int]] = {}
    for domain, counter in votes.items():
        pname, n = counter.most_common(1)[0]
        if n >= min_support:
            out[domain] = (pname, n)
    return out


def apply_patterns(contacts: Iterable[Contact], patterns: Dict[str, Tuple[str, int]]) -> int:
    """Guess addresses for contacts without one. Guesses stay below the default send threshold."""
    by_reg = {registrable_domain(d): (d, p) for d, p in patterns.items()}
    n = 0
    for c in contacts:
        if c.emails or not c.org_domain:
            continue
        hit = by_reg.get(registrable_domain(c.org_domain))
        if not hit:
            continue
        domain, (pname, support) = hit
        local = _patterns(c.name).get(pname)
        if local:
            conf = 0.5 if support >= 5 else 0.4
            c.add_email(EmailCandidate(address=f"{local}@{domain}", confidence=conf, method="pattern"))
            c.flags.append("email_guessed")
            n += 1
    return n


# ── deliverability ───────────────────────────────────────────────────────────

_mx_cache: Dict[str, Optional[bool]] = {}


def has_mx(domain: str) -> Optional[bool]:
    """True/False if the domain can / cannot receive mail, None when it cannot be checked
    (dnspython not installed or DNS unavailable)."""
    if domain in _mx_cache:
        return _mx_cache[domain]
    result: Optional[bool]
    try:
        import dns.resolver  # type: ignore

        try:
            result = len(dns.resolver.resolve(domain, "MX", lifetime=5)) > 0
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            result = False
        except Exception:
            result = None
    except ImportError:
        result = None
    _mx_cache[domain] = result
    return result


def verify_deliverable(contacts: Iterable[Contact]) -> int:
    """Drop addresses whose domain has no MX record. Returns number removed."""
    removed = 0
    for c in contacts:
        keep: List[EmailCandidate] = []
        for e in c.emails:
            if has_mx(e.address.split("@")[1]) is False:
                removed += 1
            else:
                keep.append(e)
        c.emails = keep
    return removed
