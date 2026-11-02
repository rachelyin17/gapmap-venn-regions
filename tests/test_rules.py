"""The rules one at a time: exact matching (V1), the region of each membership triple (V2), papers left out
(V3) and the order of ids (V4), plus the checks on a dataset and a query."""

from __future__ import annotations

import pytest

from plugin.rules import REGION_OF, REGIONS, concept_problem, papers_problem, query_problem, regions


def paper(paper_id: str, *concepts: str) -> dict:
    return {"paper_id": paper_id, "concepts": sorted(concepts)}


def test_v1_matches_exact_strings_only() -> None:
    found = regions([paper("P1", "large language model"), paper("P2", "llms")], "language model", "llm", "x")
    assert all(ids == [] for ids in found.values())


@pytest.mark.parametrize(("has", "region"), sorted(REGION_OF.items()))
def test_v2_places_each_membership_triple_in_its_region(has: tuple, region: str) -> None:
    concepts = [name for name, present in zip("abc", has, strict=True) if present]
    found = regions([paper("P1", *concepts, "other")], "a", "b", "c")
    assert found == {name: ["P1"] if name == region else [] for name in REGIONS}


def test_v2_covers_the_7_regions_once_each() -> None:
    assert sorted(REGION_OF.values()) == sorted(REGIONS) and len(REGION_OF) == 7


def test_v2_follows_a_b_and_c_as_given() -> None:
    papers = [paper("P1", "x"), paper("P2", "x", "y")]
    assert regions(papers, "x", "y", "z")["a_only"] == ["P1"] and regions(papers, "y", "x", "z")["b_only"] == ["P1"]
    assert regions(papers, "z", "x", "y")["bc_only"] == ["P2"]


def test_v3_leaves_out_papers_with_none_of_the_three() -> None:
    found = regions([paper("P1"), paper("P2", "q")], "a", "b", "c")
    assert found == {name: [] for name in REGIONS}


def test_v4_sorts_ids_in_python_string_order() -> None:
    assert regions([paper("P2", "a"), paper("P10", "a"), paper("P1", "a")], "a", "b", "c")["a_only"] == ["P1", "P10", "P2"]


@pytest.mark.parametrize(
    ("concept", "words"),
    [("Bert", "outside [a-z0-9 ]"), ("gpt-4", "'-'"), (" bert", "leading"), ("a  b", "double"), ("", "non-empty"), (4, "non-empty")],
)
def test_concept_problem_refuses_an_unnormalized_concept(concept, words: str) -> None:
    assert words in concept_problem(concept)


@pytest.mark.parametrize(
    ("concepts", "papers", "words"),
    [
        (["b", "a"], [], "vocabulary concepts are not sorted ascending"),
        (["a", "a"], [], "vocabulary concept 'a' is listed more than once"),
        (["a", "B"], [], "outside [a-z0-9 ]"),
        (["a"], [paper("P1", "c")], "P1 has 'c', which is not in the vocabulary"),
        (["a", "b"], [{"paper_id": "P1", "concepts": ["b", "a"]}], "P1's concepts are not sorted ascending"),
        (["a"], [{"paper_id": "P1", "concepts": ["a", "a"]}], "P1's concept 'a' is listed more than once"),
        (["a"], [paper("P2"), paper("P10")], "paper ids are not sorted ascending: 'P2' comes before 'P10'"),
        (["a"], [paper("P1"), paper("P1")], "paper id 'P1' is listed more than once"),
        (["a"], [paper("")], "is not a non-empty string"),
    ],
)
def test_papers_problem_names_what_is_wrong(concepts: list, papers: list, words: str) -> None:
    assert words in papers_problem(concepts, papers)


def test_papers_problem_allows_empty_papers_and_unused_concepts() -> None:
    assert papers_problem(["a", "b", "unused"], [paper("P1"), paper("P10", "a", "b"), paper("P2", "a")]) is None


@pytest.mark.parametrize(
    ("query", "words"),
    [
        (("a", "a", "b"), "must be distinct"),
        (("a", "b", "b"), "must be distinct"),
        (("a", "b", "z"), "c 'z' is not in the dataset's vocabulary"),
        (("a", 3, "b"), "b must be a string"),
    ],
)
def test_query_problem_wants_three_distinct_vocabulary_concepts(query: tuple, words: str) -> None:
    assert words in query_problem(["a", "b", "c"], *query)
    assert query_problem(["a", "b", "c"], "c", "a", "b") is None
