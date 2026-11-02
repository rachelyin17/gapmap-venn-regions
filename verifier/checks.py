"""Shared verdict helpers: a result, the ordered constraint list the task verdict carries, the input checks
the tools share, and plain forms of the contract instances the rules work on."""

from contract import Dataset

from .rules import DATASETS, REGIONS

MOST = 10  # list entries a constraint's expected or actual shows


def result(valid, message, *, score=None, **details):
    return {
        "valid": valid,
        "score": float(valid) if score is None else score,
        "message": message,
        "details": details,
    }


def compact(value):
    """A value small enough for a verdict: lists cut to MOST entries, mappings to MOST keys."""
    if isinstance(value, list) and len(value) > MOST:
        return value[:MOST] + [f"… {len(value) - MOST} more"]
    if isinstance(value, dict) and len(value) > MOST:
        return {**dict(list(value.items())[:MOST]), "…": f"{len(value) - MOST} more"}
    return value


def constraint(identifier, text, ok, expected=None, actual=None, message=""):
    """One constraint's entry, always with the same keys: ok is True (holds), False (fails) or None (not
    checked: a check it needs did not hold); expected and actual where they say something, else None;
    and message, why it fails or why it was not checked ("" when it holds)."""
    return {
        "id": identifier,
        "constraint": text,
        "ok": ok,
        "expected": compact(expected),
        "actual": compact(actual),
        "message": "" if ok is True else (message or text),
    }


def summary(constraints):
    """A failed verdict's message: the first failing constraint (or, when none fails, the first not
    checked), its id and why, then how many more fail: "disjoint: paper 'E7' is in 2 regions: ab_only, abc (+1 more)"."""
    failed = [entry for entry in constraints if entry["ok"] is False]
    first = failed[0] if failed else next(entry for entry in constraints if entry["ok"] is not True)
    reason = first["message"].removesuffix(".")
    return f"{first['id']}: {reason}" + (f" (+{len(failed) - 1} more)" if len(failed) > 1 else "")


def dataset_problem(dataset):
    """What is wrong with a contract Dataset value, or None."""
    value = dataset.value if isinstance(dataset, Dataset) else dataset
    return None if value in DATASETS else f"dataset must be one of {', '.join(DATASETS)}, not {value!r}"


def shape_problem(value):
    """What is wrong with a VennInput's shape, or None: a known dataset and three distinct non-empty strings
    (whether they are in the vocabulary needs the pinned data, so verify_task checks that)."""
    problem = dataset_problem(value.dataset)
    if problem:
        return problem
    query = {"a": value.a, "b": value.b, "c": value.c}
    for name, concept in query.items():
        if type(concept) is not str or concept == "":
            return f"{name} must be a non-empty string, not {concept!r}"
    if len(set(query.values())) < 3:
        return f"a, b and c must be distinct, not {value.a!r}, {value.b!r} and {value.c!r}"
    return None


def plain_papers(papers):
    """Contract Papers as plain {paper_id, concepts}, in their order."""
    return [{"paper_id": paper.paper_id, "concepts": list(paper.concepts)} for paper in papers]


def plain_regions(regions):
    """A contract Regions as plain {name: ids}, in REGIONS order. Every name is read by its field: the
    verifier's sandbox allows no getattr."""
    found = {
        "a_only": regions.a_only,
        "b_only": regions.b_only,
        "c_only": regions.c_only,
        "ab_only": regions.ab_only,
        "ac_only": regions.ac_only,
        "bc_only": regions.bc_only,
        "abc": regions.abc,
    }
    return {name: list(found[name]) for name in REGIONS}
