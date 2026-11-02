"""The venn-regions verdict: 6 constraints, in order, on the answer key, the task, and the submitted regions.
The verdict is valid only when every one holds. Any failure among the first 5 (the answer key, the task and
the output's structure) scores 0; otherwise the score is the share of papers considered (those expected or
submitted in any region) whose submitted region, or none, is the one the rules give.

The rules are applied here to the raw papers the answer key pinned, never read from the expected output it
carries (the answer-key constraint holds only when that output agrees with them)."""

from .checks import constraint, plain_papers, plain_regions, result, shape_problem, summary
from .pin import pin_problem
from .rules import MOST_NAMED, REGIONS, regions

TEXT = {
    "answer-key": "The answer key pinned this task's dataset and query, and its papers",
    "task-input": "The task names a known dataset and three distinct concepts a, b and c from its vocabulary",
    "echo": "The output echoes the task's dataset, a, b and c, in that order",
    "form": "All 7 regions are sorted and unique, every id a known paper",
    "disjoint": "No paper is in more than one region",
    "regions-exact": "Each paper is in the region the rules give, or in none",
}
ORDER = tuple(TEXT)
GATES = ORDER[:5]  # the answer key, the task and the output's structure: any failure scores 0
NONE = "none"  # the region of a paper left out


def named(items):
    """Up to MOST_NAMED items joined by "; ", and how many more."""
    shown = "; ".join(items[:MOST_NAMED])
    return shown + (f"; and {len(items) - MOST_NAMED} more" if len(items) > MOST_NAMED else "")


def input_problem(arguments, reference):
    """What is wrong with the task, or None: its shape, then (when there is a pin to read the vocabulary
    from) whether a, b and c are vocabulary concepts."""
    problem = shape_problem(arguments)
    if problem or reference is None:
        return problem
    vocabulary = set(reference.concepts)
    for name, concept in (("a", arguments.a), ("b", arguments.b), ("c", arguments.c)):
        if concept not in vocabulary:
            return f"{name} {concept!r} is not in the dataset's vocabulary"
    return None


def form_problems(known, submitted):
    """What is wrong with each submitted region, in words: an unknown id, a repeated id, or an order other
    than ascending."""
    problems = []
    for name in REGIONS:
        ids = submitted[name]
        unknown = [paper_id for paper_id in ids if paper_id not in known]
        repeated = sorted({paper_id for paper_id in ids if ids.count(paper_id) > 1})
        if unknown:
            problems.append(f"{name} has {unknown[0]!r}, which is not a paper")
        elif repeated:
            problems.append(f"{name} lists {repeated[0]!r} more than once")
        elif ids != sorted(ids):
            first = next(index for index in range(len(ids) - 1) if ids[index] > ids[index + 1])
            problems.append(f"{name} is not sorted ascending: {ids[first]!r} comes before {ids[first + 1]!r}")
    return problems


def disjoint_problems(submitted):
    """Each paper id in more than one region, with the regions."""
    holding = {}
    for name in REGIONS:
        for paper_id in dict.fromkeys(submitted[name]):
            holding.setdefault(paper_id, []).append(name)
    return [f"paper {paper_id!r} is in {len(names)} regions: {', '.join(names)}" for paper_id, names in sorted(holding.items()) if len(names) > 1]


def region_map(found):
    """{paper id: its region} for regions {name: ids} whose ids are disjoint."""
    return {paper_id: name for name in REGIONS for paper_id in found[name]}


def mismatches(expected, submitted):
    """Each considered paper (expected or submitted in any region) whose submitted region differs from its
    expected one, sorted by id; "none" stands for no region."""
    want, got = region_map(expected), region_map(submitted)
    considered = sorted(set(want) | set(got))
    found = [
        {"paper_id": paper_id, "expected": want.get(paper_id, NONE), "submitted": got.get(paper_id, NONE)}
        for paper_id in considered
        if want.get(paper_id, NONE) != got.get(paper_id, NONE)
    ]
    return considered, found


def described(mismatch):
    return f"{mismatch['paper_id']}: expected {mismatch['expected']}, submitted {mismatch['submitted']}"


def verify_task(arguments, output, reference, workspace):
    """The verdict on a submitted VennOutput (see the module's docstring)."""
    checks = {}
    details = {}

    def put(identifier, ok, expected=None, actual=None, message=""):
        checks[identifier] = constraint(identifier, TEXT[identifier], ok, expected, actual, message)

    def skip(identifier, *needs):
        """Not checked: one of the checks it needs did not hold."""
        blocking = [need for need in needs if checks[need]["ok"] is not True]
        checks[identifier] = constraint(identifier, TEXT[identifier], None, message=f"Not checked: {' and '.join(blocking)} did not hold.")

    def hold(*identifiers):
        return all(checks[identifier]["ok"] is True for identifier in identifiers)

    # 1-2: the answer key and the task.
    if reference is None:
        put("answer-key", False, message="No answer key: pin_venn never pinned the dataset.")
    else:
        problem = pin_problem(arguments, reference)
        put("answer-key", problem is None, message=problem and problem[0].upper() + problem[1:])
    problem = input_problem(arguments, reference)
    put("task-input", problem is None, message=problem)

    # 3-5: the output's structure.
    want = {"dataset": arguments.dataset.value, "a": arguments.a, "b": arguments.b, "c": arguments.c}
    got = {"dataset": output.dataset.value, "a": output.a, "b": output.b, "c": output.c}
    wrong = [f"{key} {got[key]!r}" for key in want if got[key] != want[key]]
    put("echo", not wrong, want, got, f"The output says {' and '.join(wrong)}: copy dataset, a, b and c from the task, in order")
    submitted = plain_regions(output.regions)
    if hold("answer-key"):
        problems = form_problems({paper.paper_id for paper in reference.papers}, submitted)
        put("form", not problems, None, problems or None, named(problems))
    else:
        skip("form", "answer-key")
    problems = disjoint_problems(submitted)
    put("disjoint", not problems, None, problems or None, named(problems))

    # 6: the regions, recomputed from the pinned papers.
    if hold("answer-key", "task-input", "disjoint"):
        expected = regions(plain_papers(reference.papers), arguments.a, arguments.b, arguments.c)
        considered, found = mismatches(expected, submitted)
        details.update(matched=len(considered) - len(found), papers_considered=len(considered), mismatches=found)
        message = f"{len(found)} of {len(considered)} papers are misplaced: {named([described(item) for item in found])}"
        put(
            "regions-exact",
            not found,
            {item["paper_id"]: item["expected"] for item in found},
            {item["paper_id"]: item["submitted"] for item in found},
            message,
        )
    else:
        skip("regions-exact", "answer-key", "task-input", "disjoint")

    constraints = [checks[identifier] for identifier in ORDER]
    if not all(checks[identifier]["ok"] is True for identifier in GATES):
        return result(False, summary(constraints), score=0.0, constraints=constraints, **details)
    if checks["regions-exact"]["ok"] is not True:
        return result(False, summary(constraints), score=details["matched"] / details["papers_considered"], constraints=constraints, **details)
    message = (
        f"All {len(ORDER)} checks hold: {details['papers_considered']} papers in 7 regions for a {want['a']!r}, "
        f"b {want['b']!r}, c {want['c']!r}, on the {want['dataset']} dataset."
    )
    return result(True, message, score=1.0, constraints=constraints, **details)
