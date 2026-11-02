"""get_papers: a known dataset in; out, its vocabulary and papers: normalized, unique, sorted concepts (at
least one), and papers with unique ids sorted ascending, each with a sorted, unique list of vocabulary
concepts (possibly empty)."""

from .checks import dataset_problem, plain_papers, result
from .rules import papers_problem


def verify_input(value):
    problem = dataset_problem(value.dataset)
    return result(False, problem) if problem else result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    concepts = list(output_value.concepts)
    if not concepts:
        return result(False, "a dataset has at least one vocabulary concept")
    papers = plain_papers(output_value.papers)
    problem = papers_problem(concepts, papers)
    if problem:
        return result(False, problem)
    mentions = sum(len(paper["concepts"]) for paper in papers)
    return result(True, "the papers are well formed", concepts=len(concepts), papers=len(papers), mentions=mentions)
