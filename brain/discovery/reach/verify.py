"""
Address verification, pluggable. A name on a page proves nothing about the mailbox: before an address is trusted it is
checked, and the answer is stored on the address (`EmailCandidate.deliverability`, `verified_by`).

    MXVerifier       free: the domain must accept mail (dnspython). Can only say "undeliverable" or "unknown".
    SMTPVerifier     free, opt-in (BRAIN_SMTP_VERIFY=1): asks the mail server whether it would accept the recipient
                     (RCPT TO, no message is sent) and detects catch-all domains. Many networks block port 25 and the
                     big mailbox providers do not answer honestly: both give "unknown", never a false "deliverable".
    HunterVerifier   paid plugin (HUNTER_API_KEY): Hunter's email-verifier.

A verifier answers one of: "deliverable", "undeliverable", "catch_all" (the domain accepts everything, so the
mailbox is unproven), "unknown". Only "undeliverable" removes an address; "deliverable" from a verifier that can
prove it is what makes an address `verified` (brain/discovery/reach/tiers.py).
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import smtplib
import threading
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from brain.emailfinder import has_mx
from brain.models import Contact, EmailCandidate
from brain.textutil import FREE_MAIL

log = logging.getLogger("brain.verify")

GUESSED = ("pattern", "pattern_confirmed", "pattern_global", "pattern_prior")
DELIVERABLE, UNDELIVERABLE, CATCH_ALL, UNKNOWN = "deliverable", "undeliverable", "catch_all", "unknown"


class Verifier:
    name = ""
    proves_mailbox = False  # can this verifier say "deliverable" for a mailbox, not just a domain?

    def verdict(self, address: str) -> str:
        return UNKNOWN

    def check(self, address: str) -> Optional[bool]:
        """Legacy shape: True deliverable, False undeliverable, None unknown."""
        v = self.verdict(address)
        return True if v == DELIVERABLE else False if v == UNDELIVERABLE else None


class MXVerifier(Verifier):
    name = "mx"

    def verdict(self, address: str) -> str:
        return UNDELIVERABLE if has_mx(address.split("@")[-1]) is False else UNKNOWN

    def check(self, address: str) -> Optional[bool]:
        return has_mx(address.split("@")[-1])


# Providers that never answer RCPT probes honestly (accept everything, then bounce, or block probers).
_OPAQUE = {"gmail.com", "googlemail.com", "yahoo.com", "yahoo.in", "yahoo.co.in", "outlook.com", "hotmail.com",
           "live.com", "icloud.com", "aol.com", "protonmail.com", "proton.me", "rediffmail.com"}
_USER_UNKNOWN = re.compile(r"user|mailbox|recipient|rcpt|address|account|no such|does not exist|not exist|unknown|"
                           r"invalid|not found|unavailable|rejected", re.I)


class SMTPVerifier(Verifier):
    """RCPT TO probe. Opt-in because it connects from this machine's address to other people's mail servers."""
    name = "smtp"
    proves_mailbox = True

    def __init__(self, helo: str = "", mail_from: str = "", timeout: float = 8.0, max_checks: int = 40,
                 factory: Callable = smtplib.SMTP, mx_lookup: Optional[Callable[[str], List[str]]] = None):
        self.helo = helo or os.getenv("BRAIN_SMTP_HELO", "") or "localhost"
        self.mail_from = mail_from or os.getenv("BRAIN_SMTP_FROM", "") or "verify@localhost"
        self.timeout, self.max_checks, self.checks = timeout, max_checks, 0
        self._factory, self._mx_lookup = factory, mx_lookup or _mx_hosts
        self._catch_all: Dict[str, bool] = {}
        self._lock = threading.Lock()
        self._net_failures = 0
        self.disabled = False  # set when the network refuses probes (port 25 is blocked on most home and cloud networks)

    def _rcpt(self, mx: str, address: str) -> Tuple[int, str]:
        srv = self._factory(timeout=self.timeout)
        try:
            srv.connect(mx, 25)
            srv.ehlo(self.helo)
            srv.mail(self.mail_from)
            code, msg = srv.rcpt(address)
            return int(code), msg.decode("utf-8", "ignore") if isinstance(msg, bytes) else str(msg)
        finally:
            try:
                srv.quit()
            except Exception:
                pass

    def verdict(self, address: str) -> str:
        domain = address.rsplit("@", 1)[-1].lower()
        if domain in _OPAQUE or domain in FREE_MAIL:
            return UNKNOWN
        with self._lock:
            if self.disabled or self.checks >= self.max_checks:
                return UNKNOWN
            self.checks += 1
        try:
            hosts = self._mx_lookup(domain)
            if not hosts:
                return UNKNOWN
            mx = hosts[0]
            if domain not in self._catch_all:  # a random mailbox: if it is accepted, acceptance means nothing
                probe = f"zz{secrets.token_hex(6)}@{domain}"
                code, _ = self._rcpt(mx, probe)
                self._catch_all[domain] = 200 <= code < 300
            if self._catch_all[domain]:
                return CATCH_ALL
            code, msg = self._rcpt(mx, address)
        except (OSError, smtplib.SMTPException, TimeoutError) as e:  # port 25 blocked, refused, timed out
            log.debug("smtp probe failed for %s: %s", address, e)
            with self._lock:
                self._net_failures += 1
                if self._net_failures >= 3 and not self.disabled:
                    # Three failures in a row: this network does not let us talk to mail servers. Stop paying a full
                    # timeout per address for an answer that will always be "unknown".
                    self.disabled = True
                    log.warning("SMTP verification switched off for this run: mail servers are unreachable from here")
            return UNKNOWN
        with self._lock:
            self._net_failures = 0
        if 200 <= code < 300:
            return DELIVERABLE
        if code in (550, 551, 553) and _USER_UNKNOWN.search(msg) and not re.search(r"spam|block|reputation|policy|denied", msg, re.I):
            return UNDELIVERABLE
        return UNKNOWN  # greylisting (4xx), policy blocks, full mailboxes: never a verdict


def _mx_hosts(domain: str) -> List[str]:
    try:
        import dns.resolver  # type: ignore

        answers = sorted(dns.resolver.resolve(domain, "MX", lifetime=5), key=lambda r: r.preference)
        return [str(a.exchange).rstrip(".") for a in answers]
    except Exception:
        return []


class HunterVerifier(Verifier):
    """https://hunter.io/api-documentation#email-verifier. Each check costs a Hunter credit; capped per campaign."""
    name = "hunter"
    proves_mailbox = True

    def __init__(self, fetcher, key: str, max_checks: int = 15):
        self.fetcher, self.key, self.max_checks, self.checks = fetcher, key, max_checks, 0

    def verdict(self, address: str) -> str:
        from brain.discovery.reach.providers import _count

        if self.checks >= self.max_checks or not _count(self.fetcher, "hunter_verify", self.max_checks):
            return UNKNOWN
        self.checks += 1
        data = self.fetcher.get_json("https://api.hunter.io/v2/email-verifier", params={"email": address},
                                     headers={"Authorization": f"Bearer {self.key}"})
        status = ((data or {}).get("data") or {}).get("status") if isinstance(data, dict) else None
        if status == "valid":
            return DELIVERABLE
        if status in ("invalid", "disposable"):
            return UNDELIVERABLE
        if status == "accept_all":
            return CATCH_ALL
        return UNKNOWN


def verifiers(fetcher=None) -> List[Verifier]:
    out: List[Verifier] = [MXVerifier()]
    if os.getenv("BRAIN_SMTP_VERIFY", "").lower() in ("1", "true", "yes"):
        if os.getenv("BRAIN_SMTP_FROM", "") and "@" in os.getenv("BRAIN_SMTP_FROM", ""):
            out.append(SMTPVerifier())
        else:  # servers refuse a probe that does not name a real sender: without one the checks would all be wasted
            log.warning("BRAIN_SMTP_VERIFY is set but BRAIN_SMTP_FROM (a real address you control) is not: SMTP checks skipped")
    key = os.getenv("HUNTER_API_KEY", "")
    if key and fetcher is not None:
        out.append(HunterVerifier(fetcher, key))
    return out


def _decide(address: str, vs: List[Verifier]) -> Tuple[str, str]:
    """(verdict, verifier name) from every verifier: undeliverable beats everything; then deliverable; then catch-all."""
    results = [(v.verdict(address), v.name) for v in vs]
    for want in (UNDELIVERABLE, DELIVERABLE, CATCH_ALL):
        for verdict, name in results:
            if verdict == want:
                return verdict, name
    return UNKNOWN, ""


def verify_addresses(contacts: Iterable[Contact], fetcher=None, limit: int = 40,
                     vs: Optional[List[Verifier]] = None) -> dict:
    """Check each contact's best address (and, if it is undeliverable, the next ones): store the verdict on the
    address and remove the undeliverable. Shared inboxes are checked too. Returns counts for the funnel."""
    vs = vs if vs is not None else verifiers(fetcher)
    stats = {"checked": 0, "deliverable": 0, "undeliverable": 0, "catch_all": 0, "unknown": 0}
    for c in list(contacts)[:limit]:
        for _ in range(3):  # best address; if it is dead, the runner-up
            best = c.best_email
            if best is None or best.deliverability:
                break
            verdict, by = _decide(best.address, vs)
            best.deliverability, best.verified_by = verdict, by
            stats["checked"] += 1
            stats[verdict] += 1
            if verdict != UNDELIVERABLE:
                break
            c.emails = [e for e in c.emails if e.address != best.address]
        if not any(e.method in GUESSED for e in c.emails):
            c.flags = [f for f in c.flags if f != "email_guessed"]
    return stats


def verify_guesses(contacts: Iterable[Contact], fetcher=None, limit: int = 30) -> dict:
    """Drop guessed addresses any verifier calls undeliverable; mark the ones a mailbox-proving verifier confirmed."""
    stats = {"checked": 0, "removed": 0, "confirmed": 0}
    vs = verifiers(fetcher)
    for c in list(contacts)[:limit]:
        keep: List[EmailCandidate] = []
        for e in c.emails:
            if e.method not in GUESSED:
                keep.append(e)
                continue
            stats["checked"] += 1
            verdict, by = _decide(e.address, vs)
            if verdict == UNDELIVERABLE:
                stats["removed"] += 1
                continue
            e.deliverability, e.verified_by = verdict, by
            if verdict == DELIVERABLE and by != "mx":
                e.confidence = max(e.confidence, 0.75)
                stats["confirmed"] += 1
            keep.append(e)
        c.emails = keep
        if not any(e.method in GUESSED for e in c.emails):
            c.flags = [f for f in c.flags if f != "email_guessed"]
    return stats
