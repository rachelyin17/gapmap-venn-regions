"""pin_venn, the answer key: a dataset's vocabulary and papers, the query, and the output the rules give. It
refuses invalid data, or query concepts that are not distinct or not in the vocabulary."""

from __future__ import annotations

from pathlib import Path

from contract import PinOutput, Regions, VennInput, VennOutput
from plugin.datasets import load, paper_list
from plugin.rules import query_problem, regions


def pin_venn(given: VennInput, workspace: Path) -> PinOutput:
    """The workspace is unused: the data is bundled, and the tool writes nothing."""
    concepts, papers = load(given.dataset)
    problem = query_problem(concepts, given.a, given.b, given.c)
    if problem:
        raise ValueError(problem)
    query = {"dataset": given.dataset, "a": given.a, "b": given.b, "c": given.c}
    return PinOutput(
        **query,
        concepts=list(concepts),
        papers=paper_list(papers),
        expected=VennOutput(**query, regions=Regions(**regions(papers, given.a, given.b, given.c))),
    )
