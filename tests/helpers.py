"""A task, its answer key, its gold output and the verdict on an output, as the host would run them."""

from __future__ import annotations

import copy
from pathlib import Path

from slimconfig import build_output, plain_output

import verifier
from contract import Dataset, PinOutput, VennInput, VennOutput
from plugin import pin_venn

TASK = "venn-regions"
DATASETS = [dataset.value for dataset in Dataset]
# The README's suggested queries, two per dataset, as (a, b, c): the RUSH Lab runs use these.
QUERIES = {
    "small": [("deep learning", "nlp", "computer vision"), ("nlp", "speech", "deep learning")],
    "standard": [
        ("large language model", "question answering", "retrieval augmented generation"),
        ("instruction tuning", "reinforcement learning", "large language model"),
    ],
    "edge-cases": [("large language model", "retrieval", "reasoning"), ("llm", "transformer", "quantum computing")],
}
WORKSPACE = Path(".")  # the tools take the workspace and leave it untouched
PINNED = object()  # the verdict's default reference: the task's own answer key


def task_for(dataset: str, a: str, b: str, c: str) -> dict:
    return {"dataset": dataset, "a": a, "b": b, "c": c}


def query(dataset: str, index: int) -> dict:
    return task_for(dataset, *QUERIES[dataset][index])


def pin_for(task: dict):
    return pin_venn(build_output(task, VennInput), WORKSPACE)


def gold(task: dict) -> dict:
    return plain_output(pin_for(task).expected)


def verdict(task: dict, output: dict, reference=PINNED, workspace: str = ".") -> dict:
    """The verdict, with every value built from plain JSON as the verifier's sandbox builds it."""
    reference = build_output(plain_output(pin_for(task)), PinOutput) if reference is PINNED else reference
    return verifier.verify_task(TASK, build_output(task, VennInput), build_output(output, VennOutput), reference, workspace)


def with_regions(output: dict, **changes: list[str]) -> dict:
    """``output`` with the named regions replaced."""
    changed = copy.deepcopy(output)
    changed["regions"].update(changes)
    return changed


def failed(found: dict) -> list[str]:
    return [entry["id"] for entry in found["details"]["constraints"] if entry["ok"] is False]


def unchecked(found: dict) -> list[str]:
    return [entry["id"] for entry in found["details"]["constraints"] if entry["ok"] is None]
