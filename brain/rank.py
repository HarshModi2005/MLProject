"""
Filtering and cheap pre-ranking. No LLM, no network.

Hard constraints (geography, exclusions, do-not-contact, seniority) are filters, not weights:
a candidate either qualifies or does not. What is left is ordered by a relevance pre-score so
that the expensive stages only ever see the most promising candidates.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from brain.models import CampaignSpec, Contact
from brain.packs import Pack
from brain.textutil import region_terms, registrable_domain, strip_accents

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "by", "from", "at", "is", "are",
         "we", "our", "this", "that", "as", "be", "using", "based", "via", "towards", "toward", "study", "new",
         "research", "approach", "analysis", "method", "methods", "paper", "results"}


def tokenize(text: str) -> List[str]:
    toks = [t for t in _WORD.findall(strip_accents(text or "").lower()) if t not in _STOP and len(t) > 1]
    # crude stemming so "networks" matches "network"
    return [t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t for t in toks]


# ── filters ──────────────────────────────────────────────────────────────────

def filter_contacts(
    contacts: Iterable[Contact], spec: CampaignSpec, pack: Pack,
    is_blocked: Optional[Callable[[Contact], Optional[str]]] = None, defer_region: bool = False,
) -> Tuple[List[Contact], Counter]:
    """Returns (kept, drop_reasons)."""
    contacts = list(contacts)
    kept: List[Contact] = []
    dropped: Counter = Counter()
    regions = [r.lower() for r in spec.regions]

    terms = [t for r in spec.regions for t in region_terms(r)]

    def in_region(c: Contact) -> bool:
        parts = [c.location_text, c.org]
        if c.kind == "org":  # an organisation's own description usually says where it is
            parts += [c.bio] + [ev.text + " " + (ev.detail or "") for ev in c.evidence[:4]]
        where = " ".join(x for x in parts if x).lower()
        return any(t in where for t in terms)

    # Organisations whose location is confirmed by a structured source vouch for people found on their sites.
    regional_domains = {registrable_domain(c.org_domain) for c in contacts if c.org_domain and c.location_text and in_region(c)}
    ex_orgs = [o.lower() for o in spec.exclude_organizations]
    ex_domains = {registrable_domain(d) for d in spec.exclude_domains}
    reject_roles = [r.lower() for r in pack.reject_roles]

    for c in contacts:
        if spec.countries and c.country and c.country not in spec.countries:
            dropped["country"] += 1
            continue
        if regions:
            named = c.org and any(o.lower() in c.org.lower() for o in spec.organizations)
            vouched = bool(c.org_domain) and registrable_domain(c.org_domain) in regional_domains
            if not (named or in_region(c) or vouched):
                if defer_region and c.kind == "org" and "enriched" not in c.flags:
                    c.flags.append("region_pending")  # its website has not been read yet
                else:
                    dropped["region"] += 1
                    continue
        if c.org and any(o in c.org.lower() for o in ex_orgs):
            dropped["excluded_org"] += 1
            continue
        if c.org_domain and registrable_domain(c.org_domain) in ex_domains:
            dropped["excluded_domain"] += 1
            continue
        if c.title and any(re.search(rf"\b{re.escape(r)}\b", c.title.lower()) for r in reject_roles):
            dropped["role"] += 1
            continue
        if "seniority" in c.signals and c.signals["seniority"] < pack.min_seniority:
            dropped["seniority"] += 1
            continue
        if is_blocked:
            reason = is_blocked(c)
            if reason:
                dropped[reason] += 1
                continue
        kept.append(c)
    return kept, dropped


# ── relevance ────────────────────────────────────────────────────────────────

def bm25_scores(query: str, docs: Sequence[str], k1: float = 1.4, b: float = 0.7) -> List[float]:
    """BM25 of each doc against the query, scaled to [0, 1] by the best doc."""
    tokenized = [tokenize(d) for d in docs]
    n = len(tokenized)
    if n == 0:
        return []
    avg_len = sum(len(d) for d in tokenized) / n or 1.0
    df: Counter = Counter()
    for d in tokenized:
        df.update(set(d))
    q_terms = Counter(tokenize(query))
    scores: List[float] = []
    for d in tokenized:
        tf = Counter(d)
        s = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(d) / avg_len))
        scores.append(s)
    top = max(scores) or 1.0
    return [s / top for s in scores]


def embedding_scores(query: str, docs: Sequence[str]) -> List[float]:
    """Cosine similarity with sentence-transformers (opt-in: BRAIN_EMBEDDINGS=sentence-transformers)."""
    from sentence_transformers import SentenceTransformer, util  # heavy import, only on request

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q = model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    d = model.encode(list(docs), convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
    sims = util.cos_sim(q, d)[0].tolist()
    return [max(0.0, min(1.0, s)) for s in sims]


def role_match(contact: Contact, roles: List[str]) -> float:
    if not roles:
        return 0.5
    if not contact.title:
        return 0.4
    title = contact.title.lower()
    return 1.0 if any(r.lower() in title for r in roles) else 0.15


def prescore(contacts: List[Contact], spec: CampaignSpec, pack: Pack, backend: str = "lexical") -> List[Contact]:
    """Sets Contact.prescore in [0, 1] and returns contacts sorted best-first."""
    if not contacts:
        return []
    docs = [c.match_text() for c in contacts]
    query = spec.query_text()
    sims = None
    if backend == "sentence-transformers":
        try:
            sims = embedding_scores(query, docs)
        except Exception:
            sims = None  # fall back rather than fail the campaign
    if sims is None:
        sims = bm25_scores(query, docs)

    roles = spec.persona.roles or pack.default_roles
    this_year = date.today().year
    for c, sim in zip(contacts, sims):
        s = c.signals
        # Activity: several recent matching items beats one.
        activity = min(1.0, math.log1p(s.get("matched_works", 0)) / math.log(6)) if s.get("matched_works") else 0.3
        if s.get("latest_year"):
            activity *= max(0.4, 1.0 - 0.2 * max(0, this_year - s["latest_year"]))
        lead = min(1.0, s.get("senior_positions", 0) / 2) if "senior_positions" in s else 0.5
        seniority = s.get("seniority", 0.5)
        fit = role_match(c, roles)
        c.prescore = round(0.50 * sim + 0.18 * activity + 0.12 * lead + 0.10 * seniority + 0.10 * fit, 4)
    return sorted(contacts, key=lambda c: c.prescore, reverse=True)


def diversify(contacts: List[Contact], limit: int, max_per_org: int) -> List[Contact]:
    """Take the best `limit` contacts while capping how many come from one organisation."""
    out: List[Contact] = []
    per_org: Dict[str, int] = Counter()
    for c in contacts:
        org = (c.org_ror or c.org_domain or c.org or c.id).lower()
        if per_org[org] >= max_per_org:
            continue
        per_org[org] += 1
        out.append(c)
        if len(out) >= limit:
            break
    return out
