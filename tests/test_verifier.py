"""The venn-regions verdict: the exact output holds every constraint on every dataset's suggested queries;
each kind of wrong, missing or malformed output fails with its own constraint, score and message; the
boundary cases come out as the rules say; plus the per-call checks."""

from __future__ import annotations

import json
from itertools import permutations

import pytest
from helpers import DATASETS, QUERIES, failed, gold, pin_for, query, task_for, unchecked, verdict, with_regions
from oracle import oracle
from slimconfig import build_output, plain_output

import verifier
from contract import Dataset, DatasetInput, PapersOutput, PinOutput, VennInput, VennOutput
from plugin.datasets import load
from plugin.rules import REGIONS, regions
from verifier.task import ORDER

EMPTY = {name: [] for name in REGIONS}
# The edge-cases answers, worked out by hand from the rules.
EDGE_CASES = {
    ("large language model", "retrieval", "reasoning"): {
        **EMPTY,
        "a_only": ["E1"],
        "b_only": ["E11", "E2"],  # E11 has language model, not large language model; E11 before E2
        "c_only": ["E3"],
        "ab_only": ["E4"],
        "ac_only": ["E5"],
        "bc_only": ["E6"],
        "abc": ["E7"],
    },
    ("llm", "transformer", "quantum computing"): {
        **EMPTY,
        "b_only": ["E1", "E10", "E11", "E2", "E3", "E4", "E5", "E6", "E7"],  # E10 has llms, not llm
        "ab_only": ["E12", "E9"],
    },
    # Q1's concepts in another order: the same papers, under the names a, b and c as given.
    ("reasoning", "large language model", "retrieval"): {
        **EMPTY,
        "a_only": ["E3"],
        "b_only": ["E1"],
        "c_only": ["E11", "E2"],
        "ab_only": ["E5"],
        "ac_only": ["E6"],
        "bc_only": ["E4"],
        "abc": ["E7"],
    },
    # No paper has any of them: every region is empty.
    ("graph neural network", "quantum computing", "speech recognition"): EMPTY,
}
SUGGESTED = [(dataset, index) for dataset in DATASETS for index in (0, 1)]
# Fixed triples beyond the suggested queries, for comparing the rules with the oracle.
TRIPLES = {
    "standard": [
        ("large language model", "question answering", "dense retrieval"),
        ("question answering", "large language model", "dense retrieval"),
        ("dense retrieval", "retrieval augmented generation", "question answering"),
        ("chain of thought", "large language model", "question answering"),
        ("chain of thought", "retrieval augmented generation", "large language model"),
        ("summarization", "question answering", "reinforcement learning"),
        ("reinforcement learning", "summarization", "instruction tuning"),
        ("instruction tuning", "chain of thought", "summarization"),
        ("large language model", "instruction tuning", "reinforcement learning"),
        ("reinforcement learning", "large language model", "instruction tuning"),
        ("retrieval augmented generation", "chain of thought", "dense retrieval"),
        ("question answering", "summarization", "large language model"),
        ("dense retrieval", "large language model", "chain of thought"),
        ("summarization", "chain of thought", "retrieval augmented generation"),
        ("instruction tuning", "dense retrieval", "question answering"),
        ("large language model", "reinforcement learning", "summarization"),
        ("question answering", "chain of thought", "instruction tuning"),
        ("retrieval augmented generation", "large language model", "question answering"),
        ("chain of thought", "dense retrieval", "reinforcement learning"),
        ("summarization", "instruction tuning", "dense retrieval"),
    ],
    "edge-cases": [
        ("language model", "large language model", "llm"),
        ("llm", "llms", "language model"),
        ("llms", "llm", "transformer"),
        ("transformer", "retrieval", "reasoning"),
        ("reasoning", "transformer", "large language model"),
        ("vision", "llms", "transformer"),
        ("graph neural network", "transformer", "vision"),
        ("quantum computing", "speech recognition", "llm"),
        ("retrieval", "reasoning", "large language model"),
        ("large language model", "reasoning", "retrieval"),
        ("retrieval", "large language model", "reasoning"),
        ("language model", "retrieval", "llm"),
        ("speech recognition", "vision", "graph neural network"),
        ("transformer", "llm", "llms"),
        ("large language model", "language model", "transformer"),
        ("reasoning", "llm", "vision"),
        ("llms", "vision", "quantum computing"),
        ("retrieval", "transformer", "language model"),
        ("llm", "reasoning", "retrieval"),
        ("vision", "graph neural network", "large language model"),
    ],
}
Q1 = query("edge-cases", 0)
Q2 = query("edge-cases", 1)
GNN_QC_SR = task_for("edge-cases", "graph neural network", "quantum computing", "speech recognition")


def wrong(task: dict, output: dict) -> dict:
    """The verdict on a well-formed but misplaced output: only regions-exact fails."""
    found = verdict(task, output)
    assert failed(found) == ["regions-exact"] and unchecked(found) == [], found["message"]
    assert 0 <= found["score"] < 1 and not found["valid"]
    return found


def malformed(task: dict, output: dict, constraint: str) -> dict:
    found = verdict(task, output)
    assert constraint in failed(found) and found["score"] == 0.0 and not found["valid"], found["message"]
    return found


# Correct.


@pytest.mark.parametrize(("dataset", "index"), SUGGESTED)
def test_the_exact_output_holds_every_constraint(dataset: str, index: int) -> None:
    task = query(dataset, index)
    found = verdict(task, gold(task))
    assert found["valid"] and found["score"] == 1.0, found["message"]
    assert found["message"].startswith(f"All {len(ORDER)} checks hold")
    assert [entry["ok"] for entry in found["details"]["constraints"]] == [True] * len(ORDER)
    assert found["details"]["matched"] == found["details"]["papers_considered"] > 0 and found["details"]["mismatches"] == []
    assert len(json.dumps(found)) < 8000, "the verdict lands in the chat's context: keep it small"


@pytest.mark.parametrize(("dataset", "index"), SUGGESTED)
def test_the_answer_key_agrees_with_the_set_operations_oracle(dataset: str, index: int) -> None:
    pin = plain_output(pin_for(query(dataset, index)))
    assert pin["expected"]["regions"] == oracle(pin["papers"], pin["a"], pin["b"], pin["c"])


def test_the_rules_and_the_oracle_agree_on_every_ordered_triple_of_small() -> None:
    concepts, papers = load(Dataset.SMALL)
    triples = list(permutations(concepts, 3))
    assert len(triples) == 60
    for a, b, c in triples:
        assert regions(papers, a, b, c) == oracle(papers, a, b, c), (a, b, c)


@pytest.mark.parametrize("dataset", ["standard", "edge-cases"])
def test_the_rules_and_the_oracle_agree_on_20_fixed_triples(dataset: str) -> None:
    concepts, papers = load(Dataset(dataset))
    triples = TRIPLES[dataset]
    assert len(set(triples)) == 20 and all(len(set(triple)) == 3 and set(triple) <= set(concepts) for triple in triples)
    for a, b, c in triples:
        assert regions(papers, a, b, c) == oracle(papers, a, b, c), (a, b, c)


@pytest.mark.parametrize("triple", list(EDGE_CASES))
def test_the_edge_cases_answers_are_the_hand_worked_ones(triple: tuple) -> None:
    task = task_for("edge-cases", *triple)
    assert gold(task)["regions"] == EDGE_CASES[triple]
    assert verdict(task, gold(task))["valid"]


def test_the_two_edge_cases_queries_fill_every_region() -> None:
    first, second = (EDGE_CASES[QUERIES["edge-cases"][index]] for index in (0, 1))
    assert all(first[name] or second[name] for name in REGIONS)
    assert first["abc"] and not second["abc"]
    assert not any(second[name] for name in ("c_only", "ac_only", "bc_only", "abc")), "quantum computing is in no paper"


def test_region_names_follow_a_b_and_c_as_given() -> None:
    given, reordered = EDGE_CASES[QUERIES["edge-cases"][0]], EDGE_CASES[("reasoning", "large language model", "retrieval")]
    assert given != reordered
    assert sorted(i for ids in given.values() for i in ids) == sorted(i for ids in reordered.values() for i in ids)
    assert given["a_only"] == reordered["b_only"] and given["ab_only"] == reordered["bc_only"]


def test_an_empty_query_scores_1_only_with_every_region_empty() -> None:
    found = verdict(GNN_QC_SR, gold(GNN_QC_SR))
    assert found["valid"] and found["score"] == 1.0 and found["details"]["papers_considered"] == 0
    found = wrong(GNN_QC_SR, with_regions(gold(GNN_QC_SR), a_only=["E1"]))
    assert found["score"] == 0.0 and found["details"]["papers_considered"] == 1
    assert found["message"] == "regions-exact: 1 of 1 papers are misplaced: E1: expected none, submitted a_only"


# Incorrect: valid structure, misplaced papers; the score is the share of considered papers placed right.


def test_a_paper_with_all_three_placed_in_ab_only() -> None:
    found = wrong(Q1, with_regions(gold(Q1), ab_only=["E4", "E7"], abc=[]))
    assert found["score"] == pytest.approx(7 / 8)
    assert found["message"] == "regions-exact: 1 of 8 papers are misplaced: E7: expected abc, submitted ab_only"


def test_ab_only_and_ac_only_swapped() -> None:
    found = wrong(Q1, with_regions(gold(Q1), ab_only=["E5"], ac_only=["E4"]))
    assert found["score"] == pytest.approx(6 / 8)
    assert found["details"]["mismatches"] == [
        {"paper_id": "E4", "expected": "ab_only", "submitted": "ac_only"},
        {"paper_id": "E5", "expected": "ac_only", "submitted": "ab_only"},
    ]


def test_a_paper_with_none_of_the_three_included() -> None:
    found = wrong(Q1, with_regions(gold(Q1), a_only=["E1", "E12"]))  # E12 has language model and llm only
    assert found["score"] == pytest.approx(8 / 9) and found["details"]["papers_considered"] == 9
    assert "E12: expected none, submitted a_only" in found["message"]
    found = wrong(Q2, with_regions(gold(Q2), a_only=["E8"]))  # E8 has no concepts at all
    assert found["score"] == pytest.approx(11 / 12)


def test_a_relevant_paper_left_out() -> None:
    found = wrong(Q1, with_regions(gold(Q1), c_only=[]))
    assert found["score"] == pytest.approx(7 / 8)
    assert found["message"] == "regions-exact: 1 of 8 papers are misplaced: E3: expected c_only, submitted none"


def test_a_near_miss_string_counted_as_a_match() -> None:
    found = wrong(Q1, with_regions(gold(Q1), b_only=["E2"], ab_only=["E11", "E4"]))  # language model taken for large language model
    assert found["score"] == pytest.approx(7 / 8)
    assert "E11: expected b_only, submitted ab_only" in found["message"]
    found = wrong(Q2, with_regions(gold(Q2), b_only=["E1", "E11", "E2", "E3", "E4", "E5", "E6", "E7"], ab_only=["E10", "E12", "E9"]))  # llms for llm
    assert found["score"] == pytest.approx(10 / 11)


def test_regions_labeled_by_sorted_concept_order() -> None:
    # Sorted, Q1's concepts are large language model, reasoning, retrieval: b and c trade places.
    swapped = gold(task_for("edge-cases", "large language model", "reasoning", "retrieval"))["regions"]
    found = wrong(Q1, {**gold(Q1), "regions": swapped})
    assert found["score"] == pytest.approx(3 / 8)
    assert [item["paper_id"] for item in found["details"]["mismatches"]] == ["E11", "E2", "E3", "E4", "E5"]


# Malformed.


def test_an_unsorted_region() -> None:
    found = malformed(Q1, with_regions(gold(Q1), b_only=["E2", "E11"]), "form")
    assert failed(found) == ["form"]
    assert found["message"] == "form: b_only is not sorted ascending: 'E2' comes before 'E11'"


def test_a_duplicate_id_inside_one_region() -> None:
    found = malformed(Q1, with_regions(gold(Q1), abc=["E7", "E7"]), "form")
    assert failed(found) == ["form"]
    assert found["message"] == "form: abc lists 'E7' more than once"


def test_the_same_id_in_two_regions() -> None:
    found = malformed(Q1, with_regions(gold(Q1), ab_only=["E4", "E7"]), "disjoint")
    assert failed(found) == ["disjoint"] and unchecked(found) == ["regions-exact"]
    assert found["message"] == "disjoint: paper 'E7' is in 2 regions: ab_only, abc"


def test_an_unknown_id() -> None:
    found = malformed(Q1, with_regions(gold(Q1), a_only=["E1", "E99"]), "form")
    assert failed(found) == ["form", "regions-exact"]
    assert found["message"] == "form: a_only has 'E99', which is not a paper (+1 more)"


def test_a_missing_region_field() -> None:
    output = gold(Q1)
    del output["regions"]["abc"]
    with pytest.raises(ValueError, match="missing field"):
        build_output(output, VennOutput)
    del output["regions"]
    with pytest.raises(ValueError, match="missing field"):
        build_output(output, VennOutput)


@pytest.mark.parametrize(
    ("changes", "said"),
    [
        ({"dataset": "small"}, "dataset 'small'"),
        ({"c": "transformer"}, "c 'transformer'"),
        ({"a": "retrieval", "b": "large language model"}, "a 'retrieval' and b 'large language model'"),
    ],
)
def test_wrong_or_swapped_echo_fields(changes: dict, said: str) -> None:
    found = malformed(Q1, {**gold(Q1), **changes}, "echo")
    assert failed(found) == ["echo"]
    assert found["message"] == f"echo: The output says {said}: copy dataset, a, b and c from the task, in order"


def test_query_concepts_that_are_not_distinct() -> None:
    task = task_for("edge-cases", "llm", "llm", "reasoning")
    with pytest.raises(ValueError, match="a, b and c must be distinct"):
        pin_for(task)
    assert not verifier.verify_input("pin_venn", build_output(task, VennInput))["valid"]
    found = verdict(task, {**gold(Q2), **task}, reference=None)
    assert failed(found) == ["answer-key", "task-input"] and found["score"] == 0.0
    assert found["details"]["constraints"][1]["message"] == "a, b and c must be distinct, not 'llm', 'llm' and 'reasoning'"


def test_a_query_concept_not_in_the_vocabulary() -> None:
    task = task_for("edge-cases", "large language model", "retrieval", "llama")
    with pytest.raises(ValueError, match="c 'llama' is not in the dataset's vocabulary"):
        pin_for(task)
    assert verifier.verify_input("pin_venn", build_output(task, VennInput))["valid"], "the shape is fine: membership needs the pin"
    reference = build_output(plain_output(pin_for(Q1)), PinOutput)  # the dataset's vocabulary, pinned for another query
    found = verdict(task, {**gold(Q1), "c": "llama"}, reference=reference)
    assert failed(found) == ["answer-key", "task-input"] and found["score"] == 0.0
    assert found["details"]["constraints"][1]["message"] == "c 'llama' is not in the dataset's vocabulary"


# Partial credit: exact fractions.


def test_partial_credit_fractions() -> None:
    assert wrong(Q1, with_regions(gold(Q1), a_only=[], b_only=["E1", "E11", "E2"]))["score"] == pytest.approx(7 / 8), "1 misplaced of 8"
    assert wrong(Q1, with_regions(gold(Q1), c_only=["E3", "E9"]))["score"] == pytest.approx(8 / 9), "1 extra: 8 right of 9"
    std = query("standard", 0)
    expected = gold(std)["regions"]
    total = sum(len(ids) for ids in expected.values())
    moved = expected["abc"][0]
    found = wrong(std, with_regions(gold(std), abc=expected["abc"][1:], ab_only=sorted([*expected["ab_only"], moved])))
    assert found["score"] == pytest.approx((total - 1) / total)


def test_the_message_names_at_most_5_misplaced_papers() -> None:
    everyone = gold(Q2)["regions"]["b_only"]
    found = wrong(Q2, with_regions(gold(Q2), b_only=[], a_only=everyone))
    assert found["score"] == pytest.approx(2 / 11)
    assert found["message"].count("expected b_only, submitted a_only") == 5 and found["message"].endswith("; and 4 more")
    assert len(found["details"]["mismatches"]) == 9


# The answer key.


def test_no_answer_key_fails_and_leaves_the_rest_unchecked() -> None:
    found = verdict(Q1, gold(Q1), reference=None)
    assert failed(found) == ["answer-key"] and unchecked(found) == ["form", "regions-exact"]
    assert found["score"] == 0.0
    assert found["message"] == "answer-key: No answer key: pin_venn never pinned the dataset"


def test_an_answer_key_for_another_query_order_fails() -> None:
    reference = build_output(plain_output(pin_for(task_for("edge-cases", "retrieval", "large language model", "reasoning"))), PinOutput)
    found = verdict(Q1, gold(Q1), reference=reference)
    assert failed(found) == ["answer-key"]
    assert "The answer key is for dataset edge-cases with a 'retrieval', b 'large language model', c 'reasoning', not" in found["message"]


def test_a_tampered_expected_output_fails_the_answer_key() -> None:
    pin = plain_output(pin_for(Q1))
    pin["expected"]["regions"]["abc"] = []
    found = verdict(Q1, gold(Q1), reference=build_output(pin, PinOutput))
    assert failed(found) == ["answer-key"]
    assert "disagrees with the rules applied to its own papers" in found["message"]


def test_a_malformed_pinned_paper_fails_the_answer_key() -> None:
    pin = plain_output(pin_for(Q1))
    pin["papers"][0]["concepts"].append("zebra")
    found = verdict(Q1, gold(Q1), reference=build_output(pin, PinOutput))
    assert failed(found) == ["answer-key"]
    assert found["message"] == "answer-key: The pinned data is malformed: paper E1 has 'zebra', which is not in the vocabulary"


def test_the_score_is_recomputed_from_the_pinned_papers_not_its_expected_output() -> None:
    pin = plain_output(pin_for(Q1))
    pin["papers"] = [paper for paper in pin["papers"] if paper["paper_id"] != "E7"]
    pin["expected"]["regions"] = oracle(pin["papers"], *QUERIES["edge-cases"][0])
    found = verdict(Q1, gold(Q1), reference=build_output(pin, PinOutput))
    assert failed(found) == ["form", "regions-exact"], "E7 is no longer a paper"


# Per call: verify_input and verify_output.


def test_verify_input_checks_the_shapes() -> None:
    assert verifier.verify_input("pin_venn", build_output(Q1, VennInput))["valid"]
    assert verifier.verify_input("get_papers", build_output({"dataset": "standard"}, DatasetInput))["valid"]
    assert not verifier.verify_input("pin_venn", VennInput(dataset=Dataset.SMALL, a="", b="nlp", c="speech"))["valid"]
    with pytest.raises(ValueError, match="is not a valid Dataset"):
        build_output({"dataset": "huge"}, DatasetInput)


@pytest.mark.parametrize("dataset", DATASETS)
def test_verify_output_accepts_each_tool_output(dataset: str) -> None:
    pin = plain_output(pin_for(query(dataset, 0)))
    papers = build_output({"concepts": pin["concepts"], "papers": pin["papers"]}, PapersOutput)
    assert verifier.verify_output("get_papers", DatasetInput(dataset=Dataset(dataset)), papers, ".")["valid"]
    built = build_output(pin, PinOutput)
    assert verifier.verify_output("pin_venn", build_output(query(dataset, 0), VennInput), built, ".")["valid"]


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (lambda data: data["concepts"].reverse(), "not sorted ascending"),
        (lambda data: data["concepts"].append("Zebra"), "outside [a-z0-9 ]"),
        (lambda data: data["concepts"].insert(0, "graph neural network"), "listed more than once"),
        (lambda data: data["papers"][0]["concepts"].append("zebra"), "'zebra', which is not in the vocabulary"),
        (lambda data: data["papers"][0]["concepts"].reverse(), "E1's concepts are not sorted"),
        (lambda data: data["papers"].reverse(), "paper ids are not sorted ascending"),
        (lambda data: data["papers"].append(dict(data["papers"][0])), "paper id 'E1' is listed more than once"),
        (lambda data: data.update(concepts=[], papers=[]), "at least one vocabulary concept"),
    ],
)
def test_verify_output_refuses_malformed_papers(change, words: str) -> None:
    pin = plain_output(pin_for(Q1))
    data = {"concepts": pin["concepts"], "papers": pin["papers"]}
    change(data)
    found = verifier.verify_output("get_papers", DatasetInput(dataset=Dataset.EDGE_CASES), build_output(data, PapersOutput), ".")
    assert not found["valid"] and words in found["message"]
