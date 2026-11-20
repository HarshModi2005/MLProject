"""
Signal registry: the kinds of proof a discovery source can attach to a candidate (Contact.signals).

A criterion in a prospect profile names a signal ("has sponsored similar events" -> past_sponsorship); a
source declares which signals it proves; the router adds a source when the campaign needs proof only that
source can give; the fit model and the judge read the tagged values. Adding a kind of proof is one entry
here (or a "declare_signals" block in a source manifest).

    proof   a direct, sourced fact about what the candidate has done (counts towards proof_rate)
    judge   shown to the LLM judge as a verified signal
    decay   days after which the signal should be considered stale (used by the persistent index)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Signal:
    name: str
    description: str
    proof: bool = True
    judge: bool = True
    decay_days: Optional[int] = None


SIGNALS: Dict[str, Signal] = {}

# Criteria may use these two without any source proving them as a signal: they are matched on text.
TEXT_SIGNALS = ("topic", "location")


def register_signal(name: str, description: str, proof: bool = True, judge: bool = True,
                    decay_days: Optional[int] = None) -> Signal:
    sig = Signal(name, description, proof, judge, decay_days)
    SIGNALS.setdefault(name, sig)
    return SIGNALS[name]


_BUILTIN: List[Tuple] = [
    # (name, description, proof, judge, decay_days) — order matters: it is the historical SIGNAL_KEYS order.
    ("past_sponsorship", "listed as a sponsor or partner of a similar event", True, True, 540),
    ("hiring", "has open roles now", True, True, 30),
    ("portfolio_fit", "a VC fund or angel investor whose portfolio includes this sector or stage (use for investors)", True, True, 365),
    ("recent_coverage", "wrote, published or covered the topic recently", True, True, 90),
    ("publications", "publishes research on the topic", True, False, 730),
    ("audience", "reaches the audience the sender wants (media, creators, brands, communities)", True, False, 365),
    ("spoke_before", "spoke, judged or mentored at a similar event", True, True, 730),
    ("local_business", "a local business (shop, clinic, school, dealer) operating in the target area — for local selling only, not a location rule", True, True, 365),
    ("funding_focus", "a foundation, CSR programme or grant-maker that funds this cause (use for grants, not for investors)", True, False, 365),
    ("open_source", "public code shows the skills", True, False, 365),
    # judge-visible measurements (not proof on their own)
    ("open_roles", "number of open roles", False, True, 30),
    ("events_count", "number of similar events sponsored", False, True, 540),
    # discovery v3
    ("app_publisher", "publishes an app or extension in a public store", True, True, 365),
    ("uses_tech", "its site or stack shows it uses a given technology", True, True, 180),
    ("recently_funded", "raised money recently", True, True, 180),
    ("open_tender", "has an open tender or request for proposals", True, True, 60),
    ("grant_awarded", "received or awarded a grant on the topic", True, True, 730),
    ("exhibits_at", "exhibited at a relevant trade show", True, True, 540),
    ("maintains_package", "maintains a published software package", True, True, 365),
    ("warm_path", "a mutual connection or an earlier conversation with the sender", True, True, None),
    ("accelerator_alum", "went through an accelerator programme", True, True, None),
    ("csr_spend", "spends CSR money on this cause", True, True, 365),
    ("member_of", "member of a relevant industry association or network", True, False, 730),
]
for _row in _BUILTIN:
    register_signal(*_row)


def proof_signals() -> Tuple[str, ...]:
    return tuple(s.name for s in SIGNALS.values() if s.proof)


def judge_signals() -> Tuple[str, ...]:
    return tuple(s.name for s in SIGNALS.values() if s.judge)


def known(name: str) -> bool:
    return name in SIGNALS or name in TEXT_SIGNALS
