# Agent evaluation applications (SYNTHETIC)

Every file here is **SYNTHETIC**. The businesses, people, FEINs (prefix `00-`, never issued by the IRS), account
numbers (prefix `99`), amounts and notes are made up by `scripts/make_applications.py` (seed 42). They stand in for
document extraction, which is Phase 2: in production the same structured fields would be extracted, with page
citations, from bank statements, tax returns, loss runs and property schedules.

- `APP-C-*.json`: small-business loan applications for the fictional Northwind Business Credit guidelines.
- `APP-I-*.json`: commercial property and liability submissions for the fictional Harborline Mutual guidelines.
- `manifest.json`: counts, the generator version, the seed, the checksum of every file and of the whole set, and
  the checksums of the watchlist files used.

Each application has a `ground_truth` block written by the generator: the scenario, the expected recommendation, the
rule codes the case is built to trigger, whether the notes contain injected instructions, and which owner (if any)
was derived from a watchlist entry. The agent never reads it (the graph's intake node drops it); only
`scripts/eval_agent.py` does.

Two kinds of adversarial case:
- **Injected instructions** in the `notes` field (7 of 65). One is phrased so that the injection detector's patterns
  do not catch it.
- **Watchlist-derived owners** (5 of 65): an organization owner whose name is a lightly varied copy of a real OFAC SDN
  *organization* entry (US government data, public domain). No individual's name from any list is used.

Made-up names that exactly equal any OFAC or OpenSanctions name are drawn again, so a "clean" owner is never a listed
name. OpenSanctions data (CC BY-NC 4.0, https://www.opensanctions.org/licensing/) is only read for that check; none
of it is written here.

Regenerate with `make applications` (needs `make data`). With the same downloaded list files the output is identical
on every run.
