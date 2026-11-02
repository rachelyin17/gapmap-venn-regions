"""The bundled datasets, data/<dataset>/papers.json, read and checked for every tool, and built into the
contract classes the tools return. rules.papers_problem finds what is wrong and this module raises it:
the verifier judges the papers the answer key pinned, and never reads these files."""

from __future__ import annotations

import json
from pathlib import Path

from contract import Dataset, Paper
from plugin.rules import papers_problem

DATA = Path(__file__).resolve().parent / "data"


def load(dataset: Dataset) -> tuple[list[str], list[dict]]:
    """A dataset's vocabulary and its papers ({paper_id, concepts}), as stored, each checked: a malformed
    file, a concept not normalized, repeated or out of order, a paper concept outside the vocabulary, or a
    paper id repeated or out of order raises."""
    data = json.loads((DATA / dataset.value / "papers.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"concepts", "papers"}:
        raise TypeError(f"{dataset.value}/papers.json must be {{concepts, papers}}")
    concepts, papers = data["concepts"], data["papers"]
    if not isinstance(concepts, list) or not isinstance(papers, list):
        raise TypeError(f"{dataset.value}/papers.json: concepts and papers must be lists")
    if not all(isinstance(paper, dict) and set(paper) == {"paper_id", "concepts"} and isinstance(paper["concepts"], list) for paper in papers):
        raise TypeError(f"{dataset.value}/papers.json: every paper must be {{paper_id, concepts}} with a list of concepts")
    problem = papers_problem(concepts, papers)
    if problem:
        raise ValueError(f"the {dataset.value} dataset: {problem}")
    return concepts, papers


def paper_list(papers: list[dict]) -> list[Paper]:
    return [Paper(paper_id=paper["paper_id"], concepts=list(paper["concepts"])) for paper in papers]
