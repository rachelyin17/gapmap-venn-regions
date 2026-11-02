"""pin_venn, the answer key: its input, and the data it pins, checked by hand. The expected output it carries
is never trusted: it must equal the rules applied to the pinned papers."""

from .checks import plain_papers, plain_regions, result, shape_problem
from .rules import papers_problem, query_problem, regions


def verify_input(value):
    problem = shape_problem(value)
    return result(False, problem) if problem else result(True, "input is valid")


def query_of(value):
    return (value.dataset, value.a, value.b, value.c)


def described_query(value):
    return f"dataset {value.dataset.value} with a {value.a!r}, b {value.b!r}, c {value.c!r}"


def pin_problem(arguments, pin):
    """What is wrong with the answer key for the task ``arguments``, or None: the task's dataset and query,
    well-formed data, a query in its vocabulary, and an expected output equal to the rules' on it."""
    if query_of(pin) != query_of(arguments):
        return f"the answer key is for {described_query(pin)}, not {described_query(arguments)}"
    concepts, papers = list(pin.concepts), plain_papers(pin.papers)
    problem = papers_problem(concepts, papers) or query_problem(concepts, pin.a, pin.b, pin.c)
    if problem:
        return f"the pinned data is malformed: {problem}"
    if query_of(pin.expected) != query_of(pin):
        return "the answer key's expected output echoes another dataset or query"
    if plain_regions(pin.expected.regions) != regions(papers, pin.a, pin.b, pin.c):
        return "the answer key's expected output disagrees with the rules applied to its own papers"
    return None


def verify_output(input_value, output_value, workspace):
    problem = pin_problem(input_value, output_value)
    if problem:
        return result(False, problem)
    placed = sum(len(ids) for ids in plain_regions(output_value.expected.regions).values())
    return result(True, "the answer key is pinned", papers=len(output_value.papers), placed=placed)
