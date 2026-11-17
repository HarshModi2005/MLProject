"""npm registry search: maintainers of packages on the topic, with the address the registry publishes for them."""

from __future__ import annotations

import re
from typing import Optional

from brain.discovery.normalize import add_published_email
from brain.models import Contact, Evidence
from brain.textutil import squash


def _owner_from_repo(url: str) -> Optional[str]:
    m = re.search(r"github\.com[/:]([A-Za-z0-9_.-]+)/", url or "")
    return m.group(1) if m else None


# Accounts that publish on behalf of projects, not people (DefinitelyTyped, CI bots, registry staff).
_BOTS = {"types", "npm", "github-actions", "dependabot", "renovate", "semantic-release-bot", "greenkeeper", "npm-support"}


def to_contact(spec, record, run) -> Optional[Contact]:
    pkg = record.data.get("package") or {}
    if str(pkg.get("name") or "").startswith("@types/"):
        return None  # type definitions, maintained by a bot account
    maintainers = [m for m in pkg.get("maintainers") or [] if m.get("username")
                   and m["username"].lower() not in _BOTS and not m["username"].lower().endswith("-bot")]
    actor = ((pkg.get("publisher") or {}).get("actor") or {})
    who = next((m for m in maintainers if m.get("email") and m["email"] == actor.get("email")), None) or \
        (maintainers[0] if maintainers else None)
    if not who:
        return None
    links = pkg.get("links") or {}
    c = Contact(kind="person", name=who["username"], title="Open-source maintainer (npm)", sources=[spec.name])
    owner = _owner_from_repo(links.get("repository") or links.get("homepage") or "")
    if owner:
        c.add_url(f"https://github.com/{owner}")
    c.add_url(links.get("npm"))
    if links.get("homepage") and "github.com" not in links["homepage"]:
        c.add_url(links["homepage"])
    c.evidence.append(Evidence(kind="publication", url=links.get("npm"),
                               text=squash(f"Maintains the npm package {pkg.get('name')}: {pkg.get('description') or ''}", 220),
                               detail=squash(f"version {pkg.get('version')} published {str(pkg.get('date') or '')[:10]}", 120)))
    add_published_email(c, who.get("email") or "", 0.75, links.get("npm"))
    c.topics = [str(k) for k in (pkg.get("keywords") or [])][:6]
    c.signals["maintains_package"] = 1.0
    c.signals["open_source"] = 0.8
    return c


def key(c: Contact) -> str:
    return c.best_email.address if c.best_email else c.name.lower()


def merge(existing: Contact, new: Contact, record) -> None:
    seen = {e.text for e in existing.evidence}
    existing.evidence += [e for e in new.evidence if e.text not in seen]
    existing.topics = list(dict.fromkeys(existing.topics + new.topics))[:10]
