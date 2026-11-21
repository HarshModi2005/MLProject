"""
What a reply says, so "no" means no.

Every first email ends with "reply 'no' and I won't write again". The reply monitor used to record any reply from a
recipient as a reply, which labels the prospect GOOD for the fit model and leaves the address free to be emailed again by
another campaign. This reads the reply (rules, no model call; free and predictable) and returns what it asks for:

    unsubscribed  stop / remove me / unsubscribe / a bare "no" / "not interested": the address is suppressed everywhere
    negative      a polite decline that is not a stop request ("not a fit for us"): marked replied, never labelled good
    auto_reply    out of office, automatic replies, delivery notices: not a reply from the person at all
    ""            anything else (positive, a question, neutral): an ordinary reply

When in doubt it chooses the safer side for the recipient: a message that both declines and asks to stop is a stop.
"""

from __future__ import annotations

import base64
import re
from typing import Any, Dict

_QUOTE_START = re.compile(r"^(on .{5,120} wrote:?|-{2,}\s*original message|from:\s.+|sent from my .+|_{5,})$", re.I)

_AUTO = re.compile(r"\b(out of (the )?office|auto-?\s?reply|automatic reply|automated (reply|response|message)|"
                   r"away from (my )?(desk|email)|on (annual )?leave|on vacation|currently (travelling|traveling|away)|"
                   r"delivery status notification|undeliverable|mail delivery (failed|subsystem)|do not reply to this)\b", re.I)

_STOP = re.compile(r"\b(unsubscribe|un-subscribe|opt[- ]?out|remove me|take me off|delete me|"
                   r"stop (emailing|e-mailing|contacting|writing|sending|messaging)|"
                   r"(do not|don'?t|dont|please do not|please don'?t|never) (email|e-mail|contact|write|message)|"
                   r"no (more|further) (emails?|e-mails?|messages?|contact)|"
                   r"(not|no longer) interested|no,? thanks?|no thank you|not for (me|us)|leave me alone|"
                   r"remove (my|this) (email|address))\b", re.I)

_BARE_NO = re.compile(r"^\W*(no|nope|stop|unsubscribe|remove)\W*$", re.I)

_DECLINE = re.compile(r"\b(not (a )?(good )?(fit|match|relevant|right)|not (looking|hiring|accepting|taking|a priority|"
                      r"something we)|(we |i )(already have|have no need)|no need|can'?t help|unable to help|pass on this|"
                      r"we'?ll pass|i'?ll pass|not at this time|not right now|no openings?|no vacanc(y|ies))\b", re.I)


def own_text(text: str) -> str:
    """The reply itself: no quoted original, no signature trail."""
    keep = []
    for line in (text or "").replace("\r", "").split("\n"):
        stripped = line.strip()
        if stripped.startswith(">"):
            continue  # a quoted line of our own email
        if _QUOTE_START.match(stripped):
            break  # everything after "On ... wrote:" is the quoted original
        keep.append(line)
    return "\n".join(keep).strip()


def classify_reply(text: str, subject: str = "", sender: str = "") -> str:
    """'unsubscribed' | 'negative' | 'auto_reply' | '' (an ordinary reply)."""
    local = (sender or "").split("@")[0].lower()
    if local in ("mailer-daemon", "postmaster", "noreply", "no-reply", "donotreply", "do-not-reply"):
        return "auto_reply"
    body = own_text(text)
    head = f"{subject}\n{body[:600]}"
    if _AUTO.search(head) and not _STOP.search(body[:300]):
        return "auto_reply"
    if _BARE_NO.match(body[:40]) or (_BARE_NO.match(subject or "") and len(body) < 40):
        return "unsubscribed"
    if _STOP.search(head):
        return "unsubscribed"
    if _DECLINE.search(body[:600]):
        return "negative"
    return ""


def message_text(message: Dict[str, Any]) -> str:
    """Plain text of a Gmail API message (payload parts, base64url); falls back to the snippet."""
    def walk(part: Dict[str, Any]) -> str:
        out = ""
        if (part.get("mimeType") or "").startswith("text/plain"):
            data = (part.get("body") or {}).get("data")
            if data:
                try:
                    out += base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")
                except Exception:
                    pass
        for sub in part.get("parts") or []:
            out += walk(sub)
        return out

    text = walk(message.get("payload") or {}).strip()
    return text or str(message.get("snippet") or "")


def header(message: Dict[str, Any], name: str) -> str:
    for h in (message.get("payload") or {}).get("headers") or []:
        if str(h.get("name", "")).lower() == name.lower():
            return str(h.get("value") or "")
    return ""


def unsubscribe_headers(sender_email: str) -> Dict[str, str]:
    """RFC 2369 List-Unsubscribe (mailto), so mail clients show an unsubscribe action. The request comes back as a message
    with "unsubscribe" in the subject, which classify_reply turns into a suppression."""
    sender_email = (sender_email or "").strip()
    if "@" not in sender_email:
        return {}
    return {"List-Unsubscribe": f"<mailto:{sender_email}?subject=unsubscribe>"}
