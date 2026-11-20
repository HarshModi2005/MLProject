"""
Precision layer: keep the expensive stages (enrichment searches, the judge) for candidates that can fit.

    orgtype.py    what kind of organisation a candidate is (or works at); off-type ones are demoted
    prejudge.py   one cheap batched LLM pass on the top of the ranking, before enrichment
    fitmodel.py   criterion weights and the "qualified" threshold learned from judge verdicts and labels
    priors.py     which sources paid off in earlier campaigns of the same playbook
"""
