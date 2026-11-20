"""
Source priors: what each source produced in earlier campaigns of the same playbook, per search call.

Within one campaign the explorer's bandit has a ~25-search budget, too little to learn which source pays
off. Across campaigns there is plenty: the per-source funnel (passed judge, reachable) and ledger (search
calls) of every finished campaign are folded into Store.record_source_stats, and the explorer starts from
them: the paid seed sources run best-first, and the UCB bandit adds the history as pseudo-observations.
"""

from __future__ import annotations

from typing import Dict, List

PRIOR_WEIGHT = 1.0  # one past campaign's average counts like one run in this campaign
MIN_CAMPAIGNS = 2


def yield_per_search(p: Dict[str, float]) -> float:
    """Judge-passed candidates per search call (reachable ones count a third, for campaigns without a judge)."""
    good = p.get("passed", 0.0) + p.get("reachable", 0.0) / 3
    return good / (p.get("search_calls", 0.0) + 1.0)


def order_seed_tools(paid: List[str], priors: Dict[str, Dict[str, float]]) -> List[str]:
    """Paid seed sources best-first by history, but only when every one of them has enough history to compare."""
    if not paid or any(priors.get(t, {}).get("campaigns", 0) < MIN_CAMPAIGNS for t in paid):
        return paid
    return sorted(paid, key=lambda t: yield_per_search(priors[t]), reverse=True)


def pseudo(prior: Dict[str, float]) -> Dict[str, float]:
    """A past campaign's average as (runs, qualified, cost) pseudo-observations for the bandit."""
    if not prior or prior.get("campaigns", 0) < 1:
        return {}
    c = prior["campaigns"]
    return {"runs": PRIOR_WEIGHT, "qualified": PRIOR_WEIGHT * (prior.get("passed", 0) + prior.get("reachable", 0) / 3) / c,
            "cost": PRIOR_WEIGHT * prior.get("search_calls", 0) / c}
