"""Reference sentences, seeded mutations, the stats helpers and the evaluation script."""

import json

from uwcopilot.common.stats import mcnemar_exact, rate_summary, wilson_interval
from uwcopilot.provenance.evaluation import incumbent_flags, sig_checker_flags
from uwcopilot.provenance.mutations import MUTATION_TYPES, Mutation, build_mutations
from uwcopilot.provenance.reference import build_reference_sentences, section_texts


def test_reference_sentences_meet_the_target_and_are_cited() -> None:
    """At least 100 reference sentences, each ending with its own section citation."""
    references = build_reference_sentences()
    assert len(references) >= 100
    assert all(
        reference.text.rstrip(";.").endswith(f"[{reference.section_id}]")
        or f"[{reference.section_id}]" in reference.text
        for reference in references
    )


def test_mutations_are_seeded_and_change_the_sentence() -> None:
    """Same seed, same mutations; every mutation differs from its original; controls flagged as such."""
    references = build_reference_sentences()[:40]
    sections = section_texts()
    first = build_mutations(references, sections, seed=7)
    second = build_mutations(references, sections, seed=7)
    assert [m.to_json() for m in first] == [m.to_json() for m in second]
    for mutation in first:
        if mutation.mutation_type in MUTATION_TYPES:
            assert mutation.mutated != mutation.original and mutation.expect_flag
        else:
            assert not mutation.expect_flag


def test_incumbent_misses_unit_swaps_that_the_sig_checker_catches() -> None:
    """The incumbent has no units: '20 bps' passes it but not the SIG checker."""
    sections = {"NBC-4.5": "Reduce verified annual revenue by 20%."}
    mutation = Mutation(
        "MUT-1", "unit_swap", "REF-1", "NBC-4.5", "by 20% [NBC-4.5]", "by 20 bps [NBC-4.5]", "%", " bps", True
    )
    assert not incumbent_flags(mutation, sections)
    assert sig_checker_flags(mutation, sections)


def test_wilson_and_mcnemar_known_values() -> None:
    """Wilson at 0/n and n/n stays inside [0, 1]; McNemar is 1.0 with no discordant pairs."""
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    assert wilson_interval(0, 0) is None
    assert rate_summary(3, 4)["rate"] == 0.75
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == 0.03125


def test_eval_script_writes_the_expected_keys(tmp_path) -> None:
    """The evaluation script runs offline and writes every documented key, with LLM parts 'not measured'."""
    import eval_provenance

    out = tmp_path / "provenance_metrics.json"
    run = eval_provenance.RunProvenance.start("test")
    report = eval_provenance.build_report(run)
    out.write_text(json.dumps(report))
    test = report["numeric_checker"]["mutation_test"]
    assert test["regenerated_from_seed_matches_file"] is True
    assert set(test["recall_by_type"]) == set(MUTATION_TYPES)
    assert report["memo"].startswith("not measured")
    assert report["validator_adversarial"]["n_should_reject"] >= 100
    assert report["validator_adversarial"]["n_clean"] >= 50
    assert "git_sha" in report["provenance"]
