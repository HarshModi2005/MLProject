"""
core/student_context.py

Builds concise, purpose-specific student text for planning, scoring, and email generation.
Research-internship outreach: curated highlights first — never a full resume paste.
"""

from __future__ import annotations

from orchestrator.state import UserIntent


def _fallback_highlights(intent: UserIntent) -> list[str]:
    """If LLM did not produce highlights, derive a small set from structured fields."""
    out: list[str] = []
    for p in (intent.cv_key_projects or [])[:3]:
        p = str(p).strip()
        if p and len(p) < 220:
            out.append(p)
    for pub in (intent.cv_publications_list or [])[:1]:
        pub = str(pub).strip()
        if pub:
            out.append(f"Publication / writing: {pub}")
    for w in (intent.cv_work_research_adjacent or [])[:2]:
        w = str(w).strip()
        if w and len(w) < 220:
            out.append(w)
    if intent.cv_rank:
        out.append(f"Academic standing: {intent.cv_rank}")
    elif intent.cv_cgpa and intent.cv_institution:
        out.append(f"CGPA {intent.cv_cgpa} ({intent.cv_institution})")
    for a in (intent.cv_achievements or [])[:2]:
        a = str(a).strip()
        if a and len(a) < 200:
            out.append(a)
    return out[:7]


def highlights_for_intent(intent: UserIntent) -> list[str]:
    h = [str(x).strip() for x in (intent.research_internship_highlights or []) if str(x).strip()]
    if len(h) >= 4:
        return h[:7]
    merged = h + [x for x in _fallback_highlights(intent) if x not in h]
    return merged[:7]


def build_matching_profile_text(intent: UserIntent) -> str:
    """
    For planning + relevance scoring: richer than a single paragraph, still compact.
    Emphasizes research signals aligned with professor matching.
    """
    parts: list[str] = []
    domains = ", ".join(intent.research_domains) if intent.research_domains else "(not specified)"
    parts.append(f"Stated outreach targets (user): {domains}")
    if intent.timeline:
        parts.append(f"Timeline: {intent.timeline}")

    ri = [x for x in (intent.cv_research_interests or []) if x]
    if ri:
        parts.append("Research interests evident from CV: " + "; ".join(ri[:8]))

    hx = highlights_for_intent(intent)
    if hx:
        parts.append("Curated research-relevant highlights:")
        for line in hx[:7]:
            parts.append(f"  • {line}")

    rexp = [x for x in (intent.cv_research_experience or []) if x]
    if rexp:
        parts.append("Formal research / RA / lab: " + "; ".join(str(x) for x in rexp[:4]))

    work = [x for x in (intent.cv_work_research_adjacent or []) if x]
    if work:
        parts.append("Research-adjacent roles: " + "; ".join(str(x) for x in work[:3]))

    summary = (intent.user_profile_summary or "").strip()
    if summary:
        parts.append("Narrative summary: " + summary[:900] + ("…" if len(summary) > 900 else ""))

    return "\n".join(parts) if parts else summary or "No profile provided."


def build_email_internship_context(intent: UserIntent) -> str:
    """
    For cold-email generation: strict anti-dump rules + curated bullets the model may select from.
    """
    lines: list[str] = []

    domains = ", ".join(intent.research_domains) if intent.research_domains else ""
    if domains:
        lines.append(f"USER'S STATED RESEARCH TARGETS: {domains}")
    if intent.timeline:
        lines.append(f"INTERNSHIP TIMING: {intent.timeline}")

    hx = highlights_for_intent(intent)
    lines.append("")
    lines.append(
        "RESEARCH INTERNSHIP HIGHLIGHTS (pre-curated — use only 2–4 that best match THIS professor's work; "
        "ignore the rest for this email):"
    )
    for i, line in enumerate(hx, 1):
        lines.append(f"  {i}. {line}")

    lines.append("")
    lines.append(
        "NARRATIVE (supporting context — paraphrase lightly; do not copy wholesale or stack every fact):"
    )
    lines.append(intent.user_profile_summary or "(none)")

    anchor: list[str] = []
    if intent.cv_degree:
        anchor.append(intent.cv_degree)
    if intent.cv_institution:
        inst = intent.cv_institution
        if intent.cv_institution_tag:
            inst = f"{inst} ({intent.cv_institution_tag})"
        anchor.append(inst)
    if intent.cv_cgpa:
        anchor.append(f"CGPA {intent.cv_cgpa}")
    if anchor:
        lines.append("")
        lines.append("ONE-LINE ELIGIBILITY (optional opening anchor, keep ≤ 20 words if used): " + " · ".join(anchor))

    lines.append("")
    lines.append(
        "EMAIL RULES: (1) Cite at most 2 quantitative lines total (e.g. CGPA OR one exam rank OR one competition). "
        "(2) Lead with research fit (project/method/topic), not a biography. "
        "(3) No bullet lists in the email body. "
        "(4) Do not mention fest roles, generic club lists, or skills laundry lists. "
        "(5) Every claim must trace to text above."
    )

    return "\n".join(lines)
