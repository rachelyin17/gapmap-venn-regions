# GapMap Venn Regions

A NanoAgent plugin (spec `nanoagent.plugin/v0001`), with the structure of
`gapmap-concept-clusters`. GapMap shows research gaps as Venn diagrams of
concepts. Given three concepts `a`, `b` and `c`, every paper that mentions at
least one of them falls into exactly one of the diagram's 7 exclusive regions. In
this plugin's task, `venn-regions`, the agent places each relevant paper into its
region. Unlike pairwise co-occurrence counts, the regions are exclusive, each one
accounts for all three concepts at once, and the answer is the paper ids
themselves, not counts. Every rule is deterministic, so each input has exactly
one correct output, and the verifier recomputes it from the raw papers.

The plugin id is prefixed with `gapmap-` to avoid collisions on the marketplace.

## Tools

| Tool | Does | Code |
|---|---|---|
| `get_papers` | A dataset exactly as stored: `concepts`, the vocabulary (normalized, unique, sorted), and `papers` `{paper_id, concepts}`, sorted by `paper_id` | `src/plugin/get_papers.py` |
| `pin_venn` | The answer key, hidden from the agent: the dataset, `a`, `b`, `c`, the vocabulary, the papers, and the `VennOutput` the rules give. It refuses invalid data, or query concepts that are not distinct or not in the vocabulary | `src/plugin/pin_venn.py` |

Each NanoAgent tool call costs about 15 s of sandbox overhead, so the agent
gets a single tool. No tool places papers in regions for the agent; otherwise the
task would be trivial. The data is bundled, so the plugin reads no network. Its
tools never write, though the manifest grants the standard `write: [workspace]`.

### Task

| Task | Input | Runner |
|---|---|---|
| `venn-regions` | `dataset` (`small`, `standard` or `edge-cases`) and `a`, `b`, `c`: three distinct vocabulary concepts, in an order that matters | `agent`, with `get_papers`; reference `pin_venn` |

The agent makes two calls: `get_papers`, then `submit` with
`{dataset, a, b, c, regions}`. `regions` has exactly 7 lists of paper ids:
`a_only`, `b_only`, `c_only`, `ab_only`, `ac_only`, `bc_only` and `abc`. There
are no count fields; a region's count is the length of its list.

### The rules

The manifest's `instructions` restate these rules:

- **V1** A paper has a concept only if that exact string is in its `concepts`. No substring or partial match:
  `language model` does not match `large language model`, and `llm` does not match `llms`.
- **V2** A paper with at least one of `a`, `b`, `c` goes into exactly one region, decided by which of them it has:
  only `a` → `a_only`, only `b` → `b_only`, only `c` → `c_only`, `a` and `b` → `ab_only`, `a` and `c` →
  `ac_only`, `b` and `c` → `bc_only`, all three → `abc`. Region names refer to `a`, `b` and `c` as given, never
  to a sorted order.
- **V3** A paper with none of the three is in no region.
- **V4** Each region's ids are sorted ascending in Python string order (`P10` comes before `P2`); an empty region
  is `[]`.

`src/plugin/rules.py` places each paper by its membership triple `(has a, has b, has c)`, in standard-library
code with no file I/O. `verifier/rules.py` is a byte-for-byte copy, and a test enforces it.
`tests/oracle.py` takes a different route, building each concept's set of papers and computing the regions with
set operations (`a_only = A - B - C`, …). The answer key must agree with the oracle on every suggested query.
The rules must agree with it on all 60 ordered triples of `small`'s vocabulary, and on 20 fixed triples each
for `standard` and `edge-cases`.

### Datasets

Each dataset is `src/plugin/data/<dataset>/papers.json`, holding
`{concepts, papers}`. `rules.papers_problem` reports anything wrong with the
data, and `src/plugin/datasets.py` raises it as an error:
- a vocabulary concept that is not normalized (`[a-z0-9 ]`, single spaces, no leading or trailing space), is
  repeated, or is out of order;
- a paper id that is empty, repeated or out of order;
- a paper's concept that is not in the vocabulary, is repeated, or is out of order.

A vocabulary concept may appear in no paper, and a paper's list may be empty.

Each dataset has **at most 16 papers, at most 5 concepts per paper and at most
50 concept mentions in all**, and a test enforces these limits. The agent places
every paper in its reasoning, and the hosted model caps each reply at about
2,000 tokens. In the earlier GapMap plugins, larger inputs ran out of tokens
before the agent could submit.

| Dataset | Papers | Vocabulary | Mentions | Contents |
|---|---|---|---|---|
| `small` | 8 | 5 | 16 | Simple cases on `P1`–`P8`: deep learning, NLP, vision, speech |
| `standard` | 14 | 8 | 32 | Realistic NLP concepts on arXiv ids (`large language model`, `retrieval augmented generation`, `instruction tuning`, …) |
| `edge-cases` | 12 | 11 | 30 | Every boundary case below, on `E1`–`E12` |

## Suggested queries

These are the queries for the RUSH Lab runs, two per dataset. The region
counts are the lengths of the expected lists.

| Dataset | a | b | c | Placed | a_only / b_only / c_only / ab_only / ac_only / bc_only / abc |
|---|---|---|---|---|---|
| `small` | `deep learning` | `nlp` | `computer vision` | 8 | 1 / 2 / 1 / 2 / 1 / 0 / 1 |
| `small` | `nlp` | `speech` | `deep learning` | 7 | 1 / 0 / 2 / 1 / 2 / 0 / 1 |
| `standard` | `large language model` | `question answering` | `retrieval augmented generation` | 13 | 5 / 3 / 0 / 2 / 1 / 1 / 1 |
| `standard` | `instruction tuning` | `reinforcement learning` | `large language model` | 10 | 0 / 1 / 6 / 0 / 1 / 1 / 1 |
| `edge-cases` | `large language model` | `retrieval` | `reasoning` | 8 | 1 / 2 / 1 / 1 / 1 / 1 / 1 |
| `edge-cases` | `llm` | `transformer` | `quantum computing` | 11 | 0 / 9 / 0 / 2 / 0 / 0 / 0 |

### Edge-case coverage

Q1 is `large language model`, `retrieval`, `reasoning`; Q2 is `llm`, `transformer`, `quantum computing`.

| Case | Paper or query |
|---|---|
| A vocabulary concept in no paper, queried so all its regions are empty | `quantum computing` is Q2's `c`: `c_only`, `ac_only`, `bc_only` and `abc` are empty |
| A paper with all three concepts, and a query where `abc` is empty | E7 is Q1's `abc`; Q2's `abc` is empty |
| A paper with no concepts at all | E8, in no region in either query |
| A paper with other concepts but none of the three, left out | Q1 leaves out E9 (`llm`), E10 (`llms`, `vision`) and E12 (`language model`, `llm`) |
| Near-miss strings, where the query uses one and a paper has only the other | E11 has `language model`, not Q1's `large language model`, so it is `b_only`, not `ab_only`; E10 has `llms`, not Q2's `llm`, so it is `b_only`, not `ab_only` |
| Ids whose string order differs from numeric order | Q1's `b_only` is `[E11, E2]`; Q2's `b_only` starts `E1, E10, E11, E2` and its `ab_only` is `[E12, E9]` |
| A concept in every paper (every paper that has any concept) | `transformer` is in all papers but E8; as Q2's `b`, it leaves `a_only` empty |
| At least one paper in each of the 7 regions across the two queries | Q1 fills all 7: E1, E11 and E2, E3, E4, E5, E6, E7 |
| The same three concepts in another order | Test-only: `reasoning`, `large language model`, `retrieval` (Q1's concepts reordered) moves E1 to `b_only`, E3 to `a_only`, E4 to `bc_only` and so on |
| No paper has any of the three | Test-only: `graph neural network`, `quantum computing`, `speech recognition` (all unused): every region is empty |

`edge-cases` cannot have a concept in literally every paper, because E8 has no
concepts. `transformer` is in every paper that has any. `tests/test_verifier.py`
holds the hand-worked answers to both queries and both test-only queries.

### Verifier

`verifier/task.py` judges the submission against 6 constraints, in order. Each
entry of `details.constraints` is `{id, constraint, ok, expected, actual,
message}`. `ok` is `true` (holds), `false` (fails) or `null` (not checked, because a
check it needs failed).

| # | id | What it checks | Example failure message |
|---|---|---|---|
| 1 | `answer-key` | The answer key pinned this task's dataset and query, well-formed papers, and an expected output equal to the rules applied to them | No answer key: pin_venn never pinned the dataset. |
| 2 | `task-input` | A known dataset; `a`, `b`, `c` distinct strings, all in the pinned vocabulary | c 'llama' is not in the dataset's vocabulary |
| 3 | `echo` | `dataset`, `a`, `b`, `c` exactly the task's (a swap fails) | The output says a 'retrieval' and b 'large language model': copy dataset, a, b and c from the task, in order |
| 4 | `form` | All 7 regions sorted and unique, every id a known paper | b_only is not sorted ascending: 'E2' comes before 'E11' |
| 5 | `disjoint` | No paper in more than one region | paper 'E7' is in 2 regions: ab_only, abc |
| 6 | `regions-exact` | Each considered paper's submitted region, or `none`, is the expected one | 1 of 8 papers are misplaced: E7: expected abc, submitted ab_only |

How it is scored:

- **Valid** only when all 6 hold; the score is then 1.0.
- **Score 0** when any of 1 to 5 fails.
- **Otherwise** the score is the share of correctly placed papers among the papers considered: those expected in
  any region, together with those submitted in any region. "In no region" counts as a region, so a paper wrongly
  included and a paper wrongly left out both count against the score. For example, 1 misplaced paper out of 8
  scores 7/8, and 1 extra paper beside 8 correct ones scores 8/9. When no paper is considered, the score is 1,
  which happens only when every expected and submitted region is empty.
- The message names up to 5 misplaced papers, like `E7: expected ab_only, submitted abc`, with `none` for no region.
- `details` carries `matched`, `papers_considered` and `mismatches`, which lists every misplaced paper.

The 7 region fields are part of the contract. An output missing one is refused
when it is built, before the verifier sees it, and the MCP `submit` schema marks
all 7 as required. `verify_input` checks the shape of the task input (a known
dataset; `a`, `b`, `c` three distinct non-empty strings) and of `get_papers`'s input.
Vocabulary membership needs the pinned data, so `verify_task` checks it.
`verify_output` checks the shape of `get_papers`'s output, and checks the whole
answer key, recomputed, for `pin_venn`.

## Limitations

- **Exact matching depends on canonical concepts.** V1 matches strings exactly, so `llms` and `llm` are different
  concepts here. The data must already be canonicalized, which is the job of `gapmap-concept-normalize`. Run on
  raw keywords, this task would split one concept across regions.
- **Three concepts at a time.** A query is one triple. Four or more concepts would need a 15-region diagram, and
  comparing triples means running the task once per triple.
- **Datasets are small.** The agent places papers by hand within an output cap of about 2,000 tokens, so each
  dataset is limited to 16 papers, 5 concepts per paper and 50 mentions. Real GapMap corpora are far larger and
  need code, not an agent's reasoning.
- **Counts are derived, not checked separately.** The output is paper ids. The counts a diagram shows are the
  lengths of the region lists, so they are right exactly when the ids are, and no separate count is submitted or
  scored.

## Layout

```text
.
├── nanoagent.yaml   # manifest: id, runtime, permissions (no hosts; the workspace), tools, the task and its instructions
├── contract.py      # each tool's and task's typed input and output: slimconfig Output classes
├── README.md
├── LICENSE
├── pyproject.toml   # the datasets ship as package data
├── uv.lock
├── .gitignore
├── .env             # needed environment variable names (none here)
├── src/plugin/      # the MCP server (__main__.py); one module per tool (get_papers.py, pin_venn.py), each
│                    #   (input, workspace) -> output; the rules (rules.py); the bundled datasets (data/) and
│                    #   their loading (datasets.py)
├── tests/           # the verdict (test_verifier.py), the rules, the tools and server, an independent set-operations oracle
├── ui/              # renderers.js exports {}, so every output is drawn from its JSON Schema
├── product/         # the product page a catalog shows: product.yaml and icon.svg
└── verifier/        # deterministic verifier: one module per tool, the task's 6 constraints in task.py
```

## Develop and run

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```
