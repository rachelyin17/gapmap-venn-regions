"""An independent implementation of the rules, written from the task's instructions alone with set
operations (the plugin classifies each paper by its membership triple, and this imports nothing of it), so
the tests compare two readings of the same rules."""


def oracle(papers, a, b, c):
    """The 7 regions, {name: sorted paper ids}, for raw papers [{paper_id, concepts}] and the query a, b, c."""
    having = {concept: {paper["paper_id"] for paper in papers if concept in paper["concepts"]} for concept in (a, b, c)}
    big_a, big_b, big_c = having[a], having[b], having[c]
    found = {
        "a_only": big_a - big_b - big_c,
        "b_only": big_b - big_a - big_c,
        "c_only": big_c - big_a - big_b,
        "ab_only": (big_a & big_b) - big_c,
        "ac_only": (big_a & big_c) - big_b,
        "bc_only": (big_b & big_c) - big_a,
        "abc": big_a & big_b & big_c,
    }
    return {name: sorted(ids) for name, ids in found.items()}
