"""The Venn rules, V1 to V4, and the checks on a dataset and a query they need, in standard-library code on
plain values (strings, lists and dicts) with no file I/O. verifier/rules.py is a byte-for-byte copy: the
server builds the answer key with it and the verifier recomputes the answer with it."""

DATASETS = ("small", "standard", "edge-cases")
REGIONS = ("a_only", "b_only", "c_only", "ab_only", "ac_only", "bc_only", "abc")
MOST_NAMED = 5  # the misplaced papers a verdict's message names
ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789 ")
# V2: (has a, has b, has c) -> its region; a paper with none of them (V3) has no entry.
REGION_OF = {
    (True, False, False): "a_only",
    (False, True, False): "b_only",
    (False, False, True): "c_only",
    (True, True, False): "ab_only",
    (True, False, True): "ac_only",
    (False, True, True): "bc_only",
    (True, True, True): "abc",
}


def concept_problem(concept):
    """What keeps ``concept`` from being normalized, in words, or None: a non-empty string of a-z, 0-9 and
    single spaces, with no leading or trailing space."""
    if not isinstance(concept, str) or concept == "":
        return "is not a non-empty string"
    outside = sorted(set(concept) - ALLOWED)
    if outside:
        return f"has characters outside [a-z0-9 ]: {''.join(outside)!r}"
    if concept != concept.strip(" ") or "  " in concept:
        return "has a leading, trailing or double space"
    return None


def list_problem(items, what):
    """What is wrong with a list meant to be unique and sorted ascending, in words, or None."""
    repeated = sorted({item for item in items if items.count(item) > 1})
    if repeated:
        return f"{what} {repeated[0]!r} is listed more than once"
    if items != sorted(items):
        first = next(index for index in range(len(items) - 1) if items[index] > items[index + 1])
        return f"{what}s are not sorted ascending: {items[first]!r} comes before {items[first + 1]!r}"
    return None


def papers_problem(concepts, papers):
    """What is wrong with a dataset, in words, or None. ``concepts``, the vocabulary, is a list of
    normalized strings, unique and sorted; ``papers`` a list of {paper_id, concepts} with non-empty unique
    ids, sorted by id, each concept list unique, sorted and drawn from the vocabulary (possibly empty)."""
    for concept in concepts:
        problem = concept_problem(concept)
        if problem:
            return f"vocabulary concept {concept!r} {problem}"
    problem = list_problem(concepts, "vocabulary concept")
    if problem:
        return problem
    known = set(concepts)
    for paper in papers:
        if not isinstance(paper["paper_id"], str) or paper["paper_id"] == "":
            return f"paper id {paper['paper_id']!r} is not a non-empty string"
        unknown = [concept for concept in paper["concepts"] if concept not in known]
        if unknown:
            return f"paper {paper['paper_id']} has {unknown[0]!r}, which is not in the vocabulary"
        problem = list_problem(paper["concepts"], f"paper {paper['paper_id']}'s concept")
        if problem:
            return problem
    return list_problem([paper["paper_id"] for paper in papers], "paper id")


def query_problem(concepts, a, b, c):
    """What is wrong with the query a, b, c against the vocabulary ``concepts``, or None: three distinct
    strings, each in the vocabulary."""
    query = {"a": a, "b": b, "c": c}
    for name, concept in query.items():
        if not isinstance(concept, str):
            return f"{name} must be a string, not {concept!r}"
    if len(set(query.values())) < 3:
        return f"a, b and c must be distinct, not {a!r}, {b!r} and {c!r}"
    for name, concept in query.items():
        if concept not in concepts:
            return f"{name} {concept!r} is not in the dataset's vocabulary"
    return None


def regions(papers, a, b, c):
    """V1 to V4: the 7 regions, {name: sorted paper ids}, each paper placed by its membership triple
    (has a, has b, has c), matched exactly; a paper with none of them is left out."""
    found = {name: [] for name in REGIONS}
    for paper in papers:
        has = set(paper["concepts"])
        region = REGION_OF.get((a in has, b in has, c in has))
        if region:
            found[region].append(paper["paper_id"])
    return {name: sorted(ids) for name, ids in found.items()}
