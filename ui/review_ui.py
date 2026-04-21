"""
ui/review_ui.py

Streamlit application for the Supervised Execution Mode.
Allows the user to review shortlisted professors, relevance scores,
and edit/approve/reject draft emails.
"""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import streamlit as st

import config
from db.database import get_session_factory, get_session_intent_plan, init_db, list_resumable_sessions
from db.models import Email as DbEmail
from db.models import Professor as DbProfessor
from orchestrator.state import DraftVariant, EmailStatus


def _resolve_cv_path(cv_path: Optional[str]) -> Optional[str]:
    if not cv_path or not str(cv_path).strip():
        return None
    p = Path(cv_path).expanduser()
    if p.is_file():
        return str(p.resolve())
    alt = config.BASE_DIR / cv_path
    if alt.is_file():
        return str(alt.resolve())
    return None


def _load_intent_plan_dicts(session_id: Optional[str]) -> Tuple[Optional[dict], Optional[dict]]:
    ij, pj = get_session_intent_plan(session_id)
    intent_d = json.loads(ij) if ij else None
    plan_d = json.loads(pj) if pj else None
    if intent_d and plan_d:
        return intent_d, plan_d
    root = Path(__file__).resolve().parent.parent
    state_file = root / "session_state.json"
    if state_file.is_file():
        try:
            data = json.loads(state_file.read_text())
            return data.get("intent") or intent_d, data.get("plan") or plan_d
        except json.JSONDecodeError:
            pass
    return intent_d, plan_d


def _parse_breakdown_json(raw: Optional[str]) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _parse_variants_json(raw: Optional[str]) -> tuple[list[DraftVariant], int]:
    if not raw:
        return [], 0
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], 0
    vars_raw = data.get("variants") or []
    idx = int(data.get("selected_variant_index", 0))
    out: list[DraftVariant] = []
    for v in vars_raw:
        if isinstance(v, dict):
            out.append(
                DraftVariant(
                    label=str(v.get("label", "A")),
                    subject=str(v.get("subject", "")),
                    body=str(v.get("body", "")),
                )
            )
    return out, idx


def init_st():
    """Initialize Streamlit state and DB."""
    st.set_page_config(
        page_title="Outreach agent - Review Dashboard",
        page_icon="🤖",
        layout="wide",
    )
    init_db()


def fetch_pending_emails(db) -> List[DbEmail]:
    """Fetch all emails waiting for manual review."""
    return db.query(DbEmail).filter(DbEmail.status == EmailStatus.DRAFT.value).all()


def get_professor(db_session, prof_id: str) -> Optional[DbProfessor]:
    """Fetch the associated professor from the DB."""
    return db_session.query(DbProfessor).filter(DbProfessor.id == prof_id).first()


def render_dashboard():
    """Main Streamlit dashboard to review emails."""
    st.title("🤖 Autonomous Outreach - Review Dashboard")
    st.markdown(
        "Review drafted emails before they are sent. Approve, edit, or reject each draft below."
    )

    with st.sidebar:
        st.subheader("Session context")
        sessions = list_resumable_sessions(12)
        if sessions:
            labels = [f"{sid[:8]}… · {step or '?'}" for sid, step, _ in sessions]
            pick = st.selectbox("Recent DB sessions", range(len(sessions)), format_func=lambda i: labels[i])
            st.caption(
                f"Full id: `{sessions[pick][0]}` — hints intent/plan for CV when sending drafts from that run."
            )
            st.session_state["_hint_session_id"] = sessions[pick][0]
        else:
            st.caption("No resumable sessions in DB yet (or all marked done).")

    db = get_session_factory()()
    try:
        pending = fetch_pending_emails(db)

        if not pending:
            st.info("No pending draft emails to review at the moment.")
            return

        st.subheader(f"{len(pending)} Drafts Pending Review")

        for i, email in enumerate(pending):
            prof = get_professor(db, email.professor_id)

            with st.expander(
                f"{prof.name if prof else 'Unknown'} | Subj: {email.subject[:40]}...",
                expanded=(i == 0),
            ):
                if prof:
                    st.markdown(
                        f"**Institution:** {prof.institution}  |  **Department:** {prof.department or 'N/A'}  "
                        f" |  **Relevance Score:** {round(prof.relevance_score, 3)}"
                    )
                    bd = _parse_breakdown_json(getattr(prof, "relevance_breakdown_json", None))
                    if bd:
                        st.markdown("**Score breakdown**")
                        cols = st.columns(5)
                        cols[0].metric("keyword", f"{bd.get('keyword_overlap', 0):.2f}")
                        cols[1].metric("semantic", f"{bd.get('embedding_sim', 0):.2f}")
                        cols[2].metric("recency", f"{bd.get('recency', 0):.2f}")
                        cols[3].metric("geo", f"{bd.get('geographic', 0):.2f}")
                        cols[4].metric("accepting", f"{bd.get('accepting_students', 0):.2f}")
                        if bd.get("matched_terms"):
                            st.caption("Matched terms: " + ", ".join(bd["matched_terms"][:12]))
                        pubs = bd.get("publication_matches") or []
                        if pubs:
                            st.markdown("**Publication overlap**")
                            for m in pubs[:4]:
                                st.caption(f"• {m.get('title', '')[:100]} — {m.get('reason', '')}")
                    if getattr(prof, "email_confidence", None) is not None:
                        st.caption(
                            f"Email source: **{prof.email_source or 'unknown'}** "
                            f"(confidence {float(prof.email_confidence):.2f})"
                        )
                    if prof.recent_publications:
                        st.markdown("**Recent Work:** " + prof.recent_publications[:150] + "...")

                st.divider()

                variants, sel_idx = _parse_variants_json(getattr(email, "variants_json", None))
                use_subj = email.subject
                use_body = email.body
                variant_choice = sel_idx
                if variants:
                    st.markdown("**A/B variants**")
                    labels = [f"{v.label}: {v.subject[:50]}…" for v in variants]
                    variant_choice = st.radio(
                        "Choose variant to edit/send",
                        range(len(variants)),
                        format_func=lambda j: labels[j],
                        index=min(sel_idx, len(variants) - 1),
                        key=f"var_{email.id}",
                    )
                    use_subj = variants[variant_choice].subject
                    use_body = variants[variant_choice].body

                with st.form(f"form_{email.id}"):
                    default_to = (email.recipient_email or "").strip()
                    if not default_to and prof and prof.email:
                        default_to = (prof.email or "").strip()
                    edited_to = st.text_input("To (recipient email)", value=default_to)
                    edited_subj = st.text_input("Subject Line", value=use_subj)
                    edited_body = st.text_area("Email Body", value=use_body, height=300)

                    col1, col2, col3 = st.columns([1, 1, 6])

                    with col1:
                        approve = st.form_submit_button("✅ Approve", use_container_width=True)
                    with col2:
                        reject = st.form_submit_button("❌ Reject", use_container_width=True)

                    if approve:
                        to_addr = (edited_to or "").strip()
                        if not to_addr or "@" not in to_addr:
                            st.error("Enter a valid recipient email in **To** before approving.")
                            st.stop()

                        email.recipient_email = to_addr
                        email.subject = edited_subj
                        email.body = edited_body
                        if variants:
                            email.variants_json = json.dumps(
                                {
                                    "selected_variant_index": int(variant_choice),
                                    "variants": [v.model_dump() for v in variants],
                                },
                                default=str,
                            )
                        email.status = EmailStatus.APPROVED.value
                        db.commit()

                        sid = email.session_id or st.session_state.get("_hint_session_id")
                        intent_d, plan_d = _load_intent_plan_dicts(sid)
                        attach = True
                        if plan_d and isinstance(plan_d.get("email_strategy"), dict):
                            attach = plan_d["email_strategy"].get("include_cv_attachment", True)
                        cv_path = None
                        if intent_d:
                            cv_path = _resolve_cv_path(intent_d.get("cv_path"))

                        from agents.gmail_sender import GmailSender

                        sender = GmailSender()
                        success = sender.send_email(
                            to_email=to_addr,
                            subject=edited_subj,
                            message_body=edited_body,
                            attachment_path=cv_path if attach else None,
                        )

                        if success:
                            email.status = EmailStatus.SENT.value
                            db.commit()
                            st.success(
                                f"Email sent successfully to {prof.name if prof else 'Unknown'}!"
                            )
                        else:
                            st.error(
                                f"Failed to send email to {prof.name if prof else 'Unknown'}. "
                                "Check console logs and your .env credentials."
                            )

                        st.rerun()
                    elif reject:
                        email.status = EmailStatus.REJECTED.value
                        db.commit()
                        st.warning(f"Email to {prof.name if prof else 'Unknown'} Rejected.")
                        st.rerun()
    finally:
        db.close()


if __name__ == "__main__":
    init_st()
    render_dashboard()
