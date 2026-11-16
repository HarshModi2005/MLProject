"""
Fit model: what a good prospect looks like, and cheap evidence-based scoring against it. No LLM.

A campaign's ProspectProfile lists weighted criteria ("has sponsored similar events", "hiring now",
"publishes on the topic"). Each criterion names the words that would support it. Scoring a candidate
means checking, criterion by criterion, whether its evidence contains that support. The result is
interpretable (which criteria are met, from which evidence) and is used three ways:

    * during discovery, to decide whether a new candidate counts as *qualified* (so the search
      stops on quality, not on volume, and query yield is measured in qualified finds);
    * in pre-ranking, next to topic similarity;
    * in the judge prompt, so the LLM checks the same criteria the user confirmed.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional, Tuple

from brain.models import CampaignSpec, Contact, FitCriterion, ProspectProfile
from brain.playbooks import Playbook
from brain.textutil import strip_accents

QUALIFIED_THRESHOLD = 0.25  # quick fit at or above this counts as a qualified find during discovery

# Evidence a channel attaches is tagged in Contact.signals with these keys (value in [0, 1]).
SIGNAL_KEYS = ("past_sponsorship", "hiring", "portfolio_fit", "recent_coverage", "publications", "audience",
               "spoke_before", "local_business", "funding_focus", "open_source")


def _crit(d: dict) -> FitCriterion:
    return FitCriterion(name=str(d.get("name") or "fit"), description=str(d.get("description") or ""),
                        weight=float(d.get("weight") or 1.0), signal=str(d.get("signal") or "topic"),
                        terms=[str(t).lower() for t in d.get("terms") or []], must=bool(d.get("must")))


def profile_for(spec: CampaignSpec, pb: Playbook) -> ProspectProfile:
    """The campaign's profile: what intake produced, completed with the playbook's defaults.
    Always contains a topic criterion built from the campaign's topics."""
    prof = spec.profile.model_copy(deep=True) if spec.profile else ProspectProfile()
    topic_terms = [t.lower() for t in spec.topics + spec.expanded_terms if t]
    if not any(c.signal == "topic" and c.name.startswith("works on") for c in prof.criteria):
        # "any of", with the related terms: the judge read "works on A, B" as requiring both, and rejected e.g. an
        # AI journalist for not covering the exact niche product.
        named = list(dict.fromkeys(spec.topics[:3] + spec.expanded_terms[:3]))
        prof.criteria.insert(0, FitCriterion(name="works on or covers any of: " + (", ".join(named) or "the topic"),
                                             weight=1.5, signal="topic", terms=topic_terms))
    have = {c.name.lower() for c in prof.criteria}
    for d in pb.criteria:
        c = _crit(d)
        if c.name.lower() not in have:
            if not c.terms and c.signal in ("topic", "audience", "recent_coverage", "local_business"):
                c.terms = topic_terms  # "audience matches" etc. are judged on the campaign's own topics
            prof.criteria.append(c)
    kind = spec.target_kind or pb.target_kind
    if kind == "org" and (spec.regions or spec.countries) and not any(c.signal == "location" for c in prof.criteria):
        from brain.orgs import _country_names
        from brain.textutil import region_terms

        terms = [t for r in spec.regions for t in region_terms(r)]
        terms += [n for n, code in _country_names().items() if code in spec.countries]
        prof.criteria.append(FitCriterion(name="based or active in " + ", ".join(spec.regions or spec.countries),
                                          weight=0.8, signal="location", terms=terms))
    have_t = {c.name.lower() for c in prof.timing_signals}
    for d in pb.timing:
        c = _crit(d)
        if c.name.lower() not in have_t:
            prof.timing_signals.append(c)
    prof.anti_patterns = list(dict.fromkeys([a.lower() for a in prof.anti_patterns + pb.anti_patterns]))
    if not prof.decision_roles:
        prof.decision_roles = list(spec.persona.roles or pb.default_roles)
    return prof


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", strip_accents(text or "").lower()) + " "


def _term_hit(term: str, text: str) -> bool:
    t = _norm(term).strip()
    if not t:
        return False
    if " " in t:  # phrase: all words present (order-free), which tolerates "sponsors of X fest"
        return all(f" {w}" in text for w in t.split() if len(w) > 2)
    return f" {t}" in text  # prefix match on word starts: "sponsor" hits "sponsored", "sponsorship"


def criterion_support(c: FitCriterion, contact: Contact, text: Optional[str] = None) -> float:
    """[0, 1]: how strongly this candidate's evidence supports the criterion."""
    if c.signal in contact.signals and c.signal != "topic":
        tagged = max(0.0, min(1.0, contact.signals[c.signal]))
        if tagged > 0:
            return tagged  # a channel produced direct proof (1.0), or weaker proof (e.g. sponsored an event elsewhere)
    text = text if text is not None else _norm(contact.match_text() + " " + (contact.org or ""))
    if not c.terms:
        return 0.0
    hits = sum(1 for t in c.terms if _term_hit(t, text))
    if not hits:
        return 0.0
    return min(1.0, 0.45 + 0.2 * hits) if len(c.terms) > 1 else 0.8


def anti_pattern_hits(prof: ProspectProfile, contact: Contact, text: Optional[str] = None) -> List[str]:
    text = text if text is not None else _norm(" ".join([contact.name, contact.title or "", contact.bio or ""] +
                                                         [e.text for e in contact.evidence[:3]]))
    return [a for a in prof.anti_patterns if _term_hit(a, text)]


def fit_scores(contact: Contact, prof: ProspectProfile) -> Tuple[float, Dict[str, float], float]:
    """(fit in [0,1], per-criterion support, timing in [0,1]). Anti-patterns cut the fit hard."""
    text = _norm(contact.match_text() + " " + (contact.org or "") + " " + contact.name)
    per = {c.name: round(criterion_support(c, contact, text), 3) for c in prof.criteria}
    total_w = sum(c.weight for c in prof.criteria) or 1.0
    fit = sum(per[c.name] * c.weight for c in prof.criteria) / total_w
    timing = max([criterion_support(c, contact, text) for c in prof.timing_signals] or [0.0])
    if contact.signals.get("latest_year"):
        age = date.today().year - contact.signals["latest_year"]
        timing = max(timing, 1.0 if age <= 0 else 0.6 if age == 1 else 0.0)
    if anti_pattern_hits(prof, contact):
        fit *= 0.3
    return round(fit, 4), per, round(timing, 4)


def quick_fit(contact: Contact, prof: ProspectProfile) -> float:
    return fit_scores(contact, prof)[0]


def is_qualified(contact: Contact, prof: ProspectProfile, threshold: float = QUALIFIED_THRESHOLD) -> bool:
    return quick_fit(contact, prof) >= threshold


def reach_factor(contact: Contact) -> float:
    """How good the way to reach them is: a person's own address beats a team inbox beats a guess."""
    best = contact.best_email
    if not best:
        return 0.0
    if best.method in ("inbox",):
        return 0.9
    if best.method in ("site",):
        return 0.85
    if best.method == "pattern":
        return 0.6
    return 1.0


def describe(prof: ProspectProfile) -> List[str]:
    """Lines for the verification card and the judge prompt."""
    lines = [f"{c.name} (weight {c.weight:g}{', required' if c.must else ''})" for c in prof.criteria]
    lines += [f"timing: {c.name}" for c in prof.timing_signals]
    if prof.anti_patterns:
        lines.append("avoid: " + ", ".join(prof.anti_patterns[:8]))
    return lines


def promising(contact: Contact, prof: ProspectProfile) -> bool:
    """Looser than is_qualified, for candidates known only from a search snippet: some criterion has support
    and nothing marks it as a poor target."""
    if anti_pattern_hits(prof, contact):
        return False
    _, per, _ = fit_scores(contact, prof)
    topical = any(v > 0 for c, v in zip(prof.criteria, per.values()) if c.signal == "topic")
    return topical or any(v >= 0.6 for v in per.values())
