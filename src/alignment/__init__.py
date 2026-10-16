"""Alignment-training extension (DPO on silver preferences).

Heavy dependencies (torch/trl/peft) are optional and pinned in
requirements-training.txt; this package's pure-python parts (preference
construction) run in the core environment and are covered by the offline
test suite.
"""
