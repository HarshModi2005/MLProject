"""
Discovery v3: sources as plugins.

    contracts.py   SourceSpec (the manifest), Record, Policy
    signals.py     registry of fit signals (kinds of proof) that sources may tag
    registry.py    loads manifests (built-in brain/discovery/sources + ./sources) into brain.tools.TOOLS
    router.py      chooses the sources for a campaign
    normalize.py   Record -> Contact (generic field map, or a source's hooks)
    adapters/      generic readers: json_api, rss, list_page, sitemap, search, python
    hooks/         per-source code for the few things a field map cannot express
    policy.py      per-source usage rules (robots, rate limit, allowed fields)
    ledger.py      what each source cost and produced in a campaign

Adding a source is one JSON manifest (see docs/discovery-v3-plan.md §4.1); most need no Python.
"""
