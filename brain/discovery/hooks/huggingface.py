"""Hugging Face Hub: people and organisations publishing popular models on the topic."""

from __future__ import annotations

from typing import Optional

from brain.models import Contact, Evidence


def to_contact(spec, record, run) -> Optional[Contact]:
    m = record.data
    mid = m.get("modelId") or m.get("id") or ""
    if "/" not in mid:
        return None
    author = mid.split("/", 1)[0]
    c = Contact(kind="person", name=author, title="Model author (Hugging Face)", sources=[spec.name])
    c.add_url(f"https://huggingface.co/{author}")
    c.evidence.append(Evidence(kind="publication", url=f"https://huggingface.co/{mid}",
                               text=f"Publishes the model {mid} on Hugging Face",
                               detail=f"{m.get('downloads', 0):,} downloads, {m.get('likes', 0)} likes"
                                      + (f", task {m['pipeline_tag']}" if m.get("pipeline_tag") else "")))
    c.signals["maintains_package"] = 1.0
    c.signals["open_source"] = min(1.0, 0.5 + (m.get("downloads") or 0) / 200000)
    c.topics = [t for t in (m.get("tags") or []) if ":" not in t][:6]
    return c


def key(c: Contact) -> str:
    return c.name.lower()


def merge(existing: Contact, new: Contact, record) -> None:
    seen = {e.text for e in existing.evidence}
    existing.evidence += [e for e in new.evidence if e.text not in seen]
    existing.signals["open_source"] = max(existing.signals.get("open_source", 0), new.signals.get("open_source", 0))
