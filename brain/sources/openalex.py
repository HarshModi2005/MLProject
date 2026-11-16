"""
OpenAlex source: topic -> recent works -> the people who wrote them.

Why works and not author search: author search matches *names*. Searching works by topic and
aggregating their authors finds people by what they actually publish, with institution, country
and ORCID attached. A whole campaign costs roughly 10-20 API requests and zero LLM calls.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import date
from typing import Any, Dict, List, Optional

import httpx

from brain.cache import Cache
from brain.models import CampaignSpec, Contact, Evidence
from brain.settings import Settings
from brain.textutil import registrable_domain, squash

log = logging.getLogger("brain.openalex")
BASE = "https://api.openalex.org"
TTL = 14 * 86400


def _short(openalex_url: Optional[str]) -> str:
    return (openalex_url or "").rsplit("/", 1)[-1]


def _abstract(inverted: Optional[Dict[str, List[int]]], words: int = 60) -> str:
    if not inverted:
        return ""
    positions = sorted((pos, w) for w, plist in inverted.items() for pos in plist if pos < words)
    return " ".join(w for _, w in positions)


class OpenAlexSource:
    name = "openalex"

    def __init__(self, settings: Optional[Settings] = None, cache: Optional[Cache] = None,
                 client: Optional[httpx.Client] = None):
        self.settings = settings or Settings()
        self.cache = cache or Cache()
        self.client = client or httpx.Client(timeout=30, headers={"User-Agent": self.settings.user_agent})
        self.requests = 0
        self.failed_terms: List[str] = []

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        key = Cache.key(path, params)
        hit = self.cache.get("openalex", key)
        if hit is not None:
            return hit
        query = dict(params)
        if self.settings.openalex_api_key:
            query["api_key"] = self.settings.openalex_api_key
        last = ""
        data = None
        for attempt in range(3):
            self.requests += 1
            try:
                r = self.client.get(BASE + path, params=query)
            except httpx.HTTPError as e:
                last = repr(e)
            else:
                if r.status_code == 200:
                    try:
                        data = r.json()
                        break
                    except ValueError as e:  # truncated body: worth retrying
                        last = f"bad JSON: {e}"
                else:
                    last = f"HTTP {r.status_code}: {r.text[:160]}"
                    if r.status_code not in (429, 500, 502, 503, 504):
                        break
            time.sleep(1.5 * (attempt + 1))
        if data is None:
            raise RuntimeError(f"OpenAlex {path} failed: {last}")
        self.cache.set("openalex", key, data, ttl=TTL)
        return data

    def search(self, spec: CampaignSpec, max_queries: int = 6, years_back: int = 3,
               max_authors: int = 300) -> List[Contact]:
        terms = list(dict.fromkeys(spec.topics + spec.expanded_terms))[:max_queries]
        if not terms:
            return []
        filters = [f"from_publication_date:{date.today().year - years_back}-01-01"]
        # Narrow at the source: searching a whole country and filtering afterwards wastes almost
        # every result when the user asked for one region or a handful of organisations.
        inst_ids = self.target_institutions(spec)
        if inst_ids:
            filters.append("authorships.institutions.id:" + "|".join(inst_ids[:100]))
        elif spec.countries:
            filters.append("authorships.institutions.country_code:" + "|".join(spec.countries))
        inst_set = set(inst_ids)

        people: Dict[str, Dict[str, Any]] = {}
        for term in terms:
            # Title/abstract search is both more precise and much faster than full-text search.
            params = {
                "filter": ",".join(filters + [f"title_and_abstract.search:{term.replace(',', ' ')}"]),
                "per-page": 100,
                "select": "id,title,publication_year,primary_location,authorships,abstract_inverted_index",
            }
            data = None
            for attempt in range(2):
                try:
                    data = self._get("/works", params)
                    break
                except Exception as e:  # one failed topic must not sink the campaign
                    log.warning("OpenAlex query failed for %r (attempt %d): %s", term, attempt + 1, e)
            if data is None:
                self.failed_terms.append(term)
                continue
            for work in data.get("results", []):
                self._collect(work, term, spec, people, inst_set)

        # Keep the most active people before paying for detail lookups.
        ranked = sorted(people.values(), key=lambda p: (len(p["works"]) + p["senior_positions"]), reverse=True)
        ranked = ranked[:max_authors]
        details = self._author_details([p["id"] for p in ranked])
        institutions = self._institutions(list({p["inst_id"] for p in ranked if p["inst_id"]}))
        return [self._to_contact(p, details.get(p["id"], {}), institutions.get(p["inst_id"] or "", {}))
                for p in ranked]

    def target_institutions(self, spec: CampaignSpec) -> List[str]:
        """Institution ids for the regions and organisations named in the spec (empty = no narrowing)."""
        ids: List[str] = []
        country = "country_code:" + "|".join(spec.countries) + "," if spec.countries else ""
        for region in spec.regions:
            for field in ("geo.region", "geo.city"):  # "Punjab" is a region, "Boston" a city
                try:
                    data = self._get("/institutions", {"filter": f"{country}{field}:{region}", "per-page": 100,
                                                       "sort": "works_count:desc", "select": "id"})
                except Exception as e:
                    log.warning("OpenAlex region lookup failed for %r: %s", region, e)
                    continue
                found = [_short(i.get("id")) for i in data.get("results", [])]
                if found:
                    ids += found
                    break
        for org in spec.organizations[:10]:
            try:
                data = self._get("/institutions", {"search": org, "per-page": 1, "select": "id"})
            except Exception as e:
                log.warning("OpenAlex organisation lookup failed for %r: %s", org, e)
                continue
            ids += [_short(i.get("id")) for i in data.get("results", [])]
        return list(dict.fromkeys(i for i in ids if i))

    def _collect(self, work: Dict[str, Any], term: str, spec: CampaignSpec, people: Dict[str, Dict[str, Any]],
                 inst_set: Optional[set] = None):
        title = squash(work.get("title") or "", 250)
        if not title:
            return
        venue = (((work.get("primary_location") or {}).get("source")) or {}).get("display_name")
        abstract = _abstract(work.get("abstract_inverted_index"))
        authorships = work.get("authorships") or []
        for a in authorships:
            author = a.get("author") or {}
            aid = _short(author.get("id"))
            if not aid or not author.get("display_name"):
                continue
            insts = a.get("institutions") or []
            if inst_set:
                insts = [i for i in insts if _short(i.get("id")) in inst_set]
            elif spec.countries:
                insts = [i for i in insts if i.get("country_code") in spec.countries]
            if (inst_set or spec.countries) and not insts:
                continue  # co-author from outside the target institutions / countries
            inst = insts[0] if insts else {}
            p = people.setdefault(aid, {
                "id": aid, "name": author["display_name"], "orcid": author.get("orcid"),
                "inst_id": _short(inst.get("id")), "inst_name": inst.get("display_name"),
                "inst_ror": inst.get("ror"), "country": inst.get("country_code"),
                "raw_aff": "; ".join(a.get("raw_affiliation_strings") or [])[:300],
                "works": {}, "terms": set(), "senior_positions": 0,
            })
            wid = _short(work.get("id"))
            if wid not in p["works"]:
                p["works"][wid] = {"title": title, "year": work.get("publication_year"), "venue": venue,
                                   "abstract": abstract, "url": work.get("id")}
                # Last / corresponding author is usually the group lead; single-author papers count too.
                if a.get("author_position") == "last" or a.get("is_corresponding") or len(authorships) == 1:
                    p["senior_positions"] += 1
            p["terms"].add(term)

    def _author_details(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(ids), 25):
            chunk = ids[i:i + 25]
            try:
                data = self._get("/authors", {
                    "filter": "openalex:" + "|".join(chunk), "per-page": 25,
                    "select": "id,orcid,works_count,cited_by_count,summary_stats,topics,last_known_institutions",
                })
            except Exception as e:
                log.warning("OpenAlex author lookup failed: %s", e)
                continue
            for a in data.get("results", []):
                out[_short(a.get("id"))] = a
        return out

    def _institutions(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(ids), 25):
            chunk = ids[i:i + 25]
            try:
                data = self._get("/institutions", {
                    "filter": "openalex:" + "|".join(chunk), "per-page": 25,
                    "select": "id,display_name,homepage_url,country_code,geo,type",
                })
            except Exception as e:
                log.warning("OpenAlex institution lookup failed: %s", e)
                continue
            for inst in data.get("results", []):
                out[_short(inst.get("id"))] = inst
        return out

    @staticmethod
    def _to_contact(p: Dict[str, Any], detail: Dict[str, Any], inst: Dict[str, Any]) -> Contact:
        geo = inst.get("geo") or {}
        location = ", ".join(x for x in [geo.get("city"), geo.get("region"), geo.get("country")] if x)
        homepage = inst.get("homepage_url")
        c = Contact(
            name=p["name"], org=p["inst_name"], org_ror=p["inst_ror"], country=p["country"],
            org_domain=registrable_domain(homepage) if homepage else None,
            location_text="; ".join(x for x in [location, p["raw_aff"]] if x) or None,
            sources=["openalex"],
        )
        c.add_key(f"openalex:{p['id']}")
        orcid = detail.get("orcid") or p.get("orcid")
        if orcid:
            c.add_key("orcid:" + _short(orcid))
        c.add_url(f"https://openalex.org/{p['id']}")
        c.topics = [t.get("display_name") for t in (detail.get("topics") or [])[:8] if t.get("display_name")]
        works = sorted(p["works"].values(), key=lambda w: w.get("year") or 0, reverse=True)
        for w in works[:6]:
            c.evidence.append(Evidence(kind="publication", text=w["title"], url=w["url"], year=w.get("year"),
                                       detail=squash(" — ".join(x for x in [w.get("venue"), w.get("abstract")] if x), 420) or None))
        stats = detail.get("summary_stats") or {}
        c.signals = {
            "matched_works": float(len(works)),
            "matched_terms": float(len(p["terms"])),
            "senior_positions": float(p["senior_positions"]),
            "works_count": float(detail.get("works_count") or 0),
            "h_index": float(stats.get("h_index") or 0),
            "latest_year": float(max([w.get("year") or 0 for w in works] or [0])),
        }
        # Students co-author the same papers; a publication track record separates them from group leads.
        c.signals["seniority"] = round(min(1.0, math.log1p(c.signals["h_index"]) / math.log(30)), 3)
        return c
