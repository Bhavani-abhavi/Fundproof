"""Tests for uwcopilot.kyc.synthetic: noise operations, hard negatives, collisions and the stratified split."""

import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from faker import Faker
from kyc_fixtures import tiny_watchlist

from uwcopilot.kyc.normalize import normalize_name
from uwcopilot.kyc.synthetic import (
    GenerationConfig,
    apply_initials,
    apply_reorder,
    apply_transliteration,
    apply_typo,
    collision_key,
    generate_customers,
    hard_negative_organization,
    hard_negative_person,
    list_collision_keys,
    natural_order,
    noisy_variant,
    sounds_different,
    stratified_split,
)


def test_natural_order_for_ofac_people() -> None:
    assert natural_order("HASSAN, Mohammed Ali", "person") == "Mohammed Ali HASSAN"
    assert natural_order("ACME, INC.", "organization") == "ACME, INC."


def test_noise_operations_change_one_thing() -> None:
    rng = random.Random(1)
    assert apply_transliteration(["mohammed", "hassan"], random.Random(3)) != ["mohammed", "hassan"]
    typo = apply_typo(["hassan"], rng)
    assert typo is not None and typo != ["hassan"] and typo[0][0] == "h"  # first letter kept
    assert apply_initials(["mohammed", "ali", "hassan"], rng)[-1] == "hassan"  # surname never shortened
    assert sorted(apply_reorder(["ali", "hassan"], rng)) == ["ali", "hassan"]
    assert apply_reorder(["single"], rng) is None
    assert apply_typo(["abc"], rng) is None  # too short


def test_noisy_variant_is_seeded_and_records_operations() -> None:
    first = noisy_variant(["mohammed", "ali", "hassan"], "person", random.Random(7))
    second = noisy_variant(["mohammed", "ali", "hassan"], "person", random.Random(7))
    assert first == second
    name, operations = first
    assert operations and all(isinstance(operation, str) for operation in operations)
    _, org_operations = noisy_variant(["anglo", "caribbean"], "organization", random.Random(1))
    assert "honorific" not in org_operations and "initials" not in org_operations


def test_hard_negative_person_replaces_exactly_one_part_with_a_different_sound() -> None:
    fake = Faker("en_US")
    Faker.seed(42)
    original = ["mohammed", "ali", "hassan"]
    for seed in range(5):
        built = hard_negative_person(original, fake, random.Random(seed))
        if built is None:
            continue
        tokens, strategy = built
        changed = [i for i, (a, b) in enumerate(zip(original, tokens, strict=True)) if a != b]
        assert len(changed) == 1
        assert sounds_different(original[changed[0]], tokens[changed[0]])
        assert strategy == ("different_surname" if changed[0] == 2 else "different_given_name")


def test_hard_negative_organization_keeps_its_longest_word() -> None:
    tokens, strategy = hard_negative_organization(["elessa", "group"], random.Random(0))
    assert strategy == "different_words"
    assert tokens[0] == "elessa" and tokens[1] != "group"
    assert hard_negative_organization(["single"], random.Random(0)) is None


def test_collision_keys_match_regardless_of_order_and_suffixes() -> None:
    assert collision_key("Hassan Mohammed Ali", "person") == collision_key("Dr. Mohammed Ali Hassan", "person")
    keys = list_collision_keys(tiny_watchlist().entries)
    assert collision_key("Northwind Shadow Trading Ltd", "organization") in keys


def test_stratified_split_keeps_label_proportions() -> None:
    labels = ["clean"] * 100 + ["true_match"] * 10 + ["hard_negative"] * 10
    splits = stratified_split(labels, seed=42)
    counts = Counter(zip(labels, splits, strict=True))
    assert counts[("clean", "train")] == 40 and counts[("clean", "validation")] == 30
    assert counts[("true_match", "test")] == 3 and counts[("hard_negative", "train")] == 4
    assert splits == stratified_split(labels, seed=42)


def test_generate_customers_on_a_tiny_list() -> None:
    watchlist = tiny_watchlist()
    entries = watchlist.entries.copy()
    keys = list_collision_keys(entries)
    config = GenerationConfig(clean_count=40, true_match_count=4, hard_negative_count=2)
    customers, report = generate_customers(entries, keys, config)
    assert isinstance(customers, pd.DataFrame)
    assert (customers["synthetic"] == "SYNTHETIC").all()
    assert customers["customer_id"].str.startswith("SYN-KYC-").all()
    assert customers["customer_id"].is_unique
    clean = customers[customers["label"] == "clean"]
    assert len(clean) == 40
    for row in clean.itertuples():
        assert collision_key(row.name, row.entity_type) not in keys
    matches = customers[customers["label"] == "true_match"]
    assert set(matches["matched_list_entry_id"]) <= set(entries["list_entry_id"])
    assert (matches["generation"] != "").all()
    hard = customers[customers["label"] == "hard_negative"]
    assert (hard["matched_list_entry_id"] == "").all()  # a hard negative matches nothing
    assert report["clean"]["kept"] == 40
    assert set(customers["split"]) <= {"train", "validation", "test"}
    again, _ = generate_customers(entries, keys, config)
    pd.testing.assert_frame_equal(customers, again)  # same seed, same customers


def test_exact_name_homonyms_are_kept_apart_without_changing_the_customers() -> None:
    watchlist = tiny_watchlist()
    entries = watchlist.entries.copy()
    keys = list_collision_keys(entries)
    config = GenerationConfig(clean_count=40, true_match_count=4, hard_negative_count=2)
    plain, _ = generate_customers(entries, keys, config)
    # Reserve the keys of some generated clean names so they become collisions on the second run.
    reserved = set(keys) | {collision_key(name, kind) for name, kind in plain[["name", "entity_type"]].head(5).values}
    homonyms: list[dict] = []
    customers, report = generate_customers(entries, reserved, config, homonyms_out=homonyms)
    again, _ = generate_customers(entries, reserved, config)
    pd.testing.assert_frame_equal(customers, again)  # keeping homonyms draws no extra random numbers
    assert report["clean"]["distinct_exact_name_homonyms"] == len(homonyms) >= 1
    assert all(record["label"] == "exact_name_homonym" for record in homonyms)
    assert all(record["customer_id"].startswith("SYN-KYC-H-") for record in homonyms)
    assert all(collision_key(record["name"], record["entity_type"]) in reserved for record in homonyms)
    assert len({collision_key(r["name"], r["entity_type"]) for r in homonyms}) == len(homonyms)


def test_normalized_true_match_tokens_are_latin() -> None:
    watchlist = tiny_watchlist()
    customers, _ = generate_customers(
        watchlist.entries, list_collision_keys(watchlist.entries), GenerationConfig(clean_count=5, true_match_count=4)
    )
    for name in customers.loc[customers["label"] == "true_match", "name"]:
        assert all(token.isascii() for token in normalize_name(name, None).tokens)


GENERATE_IN_SUBPROCESS = """
import hashlib, sys
sys.path.insert(0, sys.argv[1])
from kyc_fixtures import tiny_watchlist
from uwcopilot.kyc.synthetic import GenerationConfig, generate_customers, list_collision_keys
entries = tiny_watchlist().entries
config = GenerationConfig(clean_count=300, true_match_count=4, hard_negative_count=2)
customers, _ = generate_customers(entries, list_collision_keys(entries), config)
print(hashlib.sha256(customers.to_csv(index=False).encode()).hexdigest())
"""


def test_generation_does_not_depend_on_pythons_hash_seed() -> None:
    """Same seed, different PYTHONHASHSEED: the customer table must be byte-identical (ground rule 3)."""
    fixtures_dir = str(Path(__file__).parent)
    digests = set()
    for hash_seed in ("1", "2"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        completed = subprocess.run(
            [sys.executable, "-c", GENERATE_IN_SUBPROCESS, fixtures_dir],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        digests.add(completed.stdout.strip())
    assert len(digests) == 1
