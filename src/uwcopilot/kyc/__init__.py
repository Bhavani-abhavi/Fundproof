"""KYC and sanctions screening (Phase 3).

Modules, in pipeline order:
    watchlist.py   load OFAC SDN (+ aliases) and OpenSanctions into one table; list snapshots (checksums)
    normalize.py   Unicode/case/punctuation folding, honorific and legal-form stripping
    countries.py   country names and codes in one comparable form
    blocking.py    token-level inverted index: Double Metaphone, exact/first token, character 3-grams
    scoring.py     token-aligned Jaro-Winkler + token-set ratio + attribute adjustment, and explanations
    reranker.py    embedding reranker (all-MiniLM-L6-v2 via fastembed/ONNX) and an offline test embedder
    matcher.py     the pipeline for one name: normalize -> block -> pre-rank -> score -> rerank -> alert
    screening.py   screen(request) -> ScreeningResult, the allow-listed "screen_entity" tool
    evaluation.py  threshold choice at an alert budget, recall-vs-alerts curves, bootstrap intervals
    synthetic.py   SYNTHETIC customers (clean, true matches, hard negatives) with ground truth
    memo.py        adverse-media memo stub: untrusted snippets, required citations, never a decision

A potential match always goes to a human reviewer. Nothing in this package clears or confirms a match.
"""
