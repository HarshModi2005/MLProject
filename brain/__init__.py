"""
brain — the vertical-agnostic discovery, ranking and drafting engine.

Pipeline (see brain/pipeline.py):

    CampaignSpec -> sources -> resolve -> filter -> pre-rank
                 -> enrich (emails) -> LLM judge (batched) -> draft

Design rule: spend network and LLM budget only on candidates that have
already survived a cheaper stage.
"""
