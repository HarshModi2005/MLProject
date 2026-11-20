"""
Per-source hooks: code for what a field map cannot express. A hooks module may define any of

    extract(spec, run, hit, page) -> [dict]     what a fetched page holds (instead of CSS selectors)
    to_contact(spec, record, run) -> Contact|None   a record as a candidate (None = skip it)
    key(contact) -> str                          identity of a candidate within this source
    merge(existing, new, record) -> None         a repeat sighting of a candidate
"""
