"""The tools as the agent and the host call them, the bundled datasets, the MCP server over stdio, and the
plugin's layout: one rules module shared byte for byte, a verifier of standard-library code, and a manifest
that restates every rule."""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import anyio
import pytest
from helpers import QUERIES
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from slimconfig import build_output, output_schema, plain_output

from contract import TOOLS, Dataset, DatasetInput, Tool, VennInput
from plugin import datasets, get_papers, pin_venn

ROOT = Path(__file__).resolve().parents[1]


def test_get_papers_gives_the_dataset_exactly_as_stored(tmp_path: Path) -> None:
    found = plain_output(get_papers(build_output({"dataset": "edge-cases"}, DatasetInput), tmp_path))
    raw = json.loads((datasets.DATA / "edge-cases" / "papers.json").read_text(encoding="utf-8"))
    assert found == raw


@pytest.mark.parametrize("dataset", list(Dataset))
def test_every_dataset_is_small_and_valid(dataset: Dataset) -> None:
    concepts, papers = datasets.load(dataset)
    # The agent places each paper in its reasoning, which the hosted model caps at about 2,000 tokens a
    # reply: at most 16 papers, 5 concepts a paper and 50 concept mentions in all.
    assert 6 <= len(papers) <= 16
    assert max(len(paper["concepts"]) for paper in papers) <= 5
    assert sum(len(paper["concepts"]) for paper in papers) <= 50
    for triple in QUERIES[dataset.value]:
        assert len(set(triple)) == 3 and set(triple) <= set(concepts), "the suggested queries are valid"


def test_the_edge_cases_dataset_has_every_boundary_case() -> None:
    concepts, papers = datasets.load(Dataset.EDGE_CASES)
    having = {concept: [paper["paper_id"] for paper in papers if concept in paper["concepts"]] for concept in concepts}
    assert having["quantum computing"] == [] and "quantum computing" in QUERIES["edge-cases"][1], "an unused concept, queried"
    assert [paper["paper_id"] for paper in papers if paper["concepts"] == []] == ["E8"], "a paper with no concepts"
    nonempty = [paper["paper_id"] for paper in papers if paper["concepts"]]
    assert having["transformer"] == nonempty, "a concept in every paper that has any"
    assert {"language model", "large language model", "llm", "llms"} <= set(concepts), "near-miss strings"
    assert [paper["paper_id"] for paper in papers][:3] == ["E1", "E10", "E11"], "string order is not numeric order"


def write_dataset(folder: Path, data: dict) -> None:
    folder.mkdir(parents=True)
    (folder / "papers.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    ("data", "error", "words"),
    [
        ({"concepts": ["a"], "papers": [{"paper_id": "P1", "concepts": ["b"]}]}, ValueError, "'b', which is not in the vocabulary"),
        ({"concepts": ["a", "a"], "papers": []}, ValueError, "listed more than once"),
        ({"concepts": ["b", "a"], "papers": []}, ValueError, "not sorted ascending"),
        ({"concepts": ["Gpt-4"], "papers": []}, ValueError, "outside [a-z0-9 ]"),
        ({"concepts": ["a"], "papers": [{"paper_id": "P2", "concepts": []}, {"paper_id": "P10", "concepts": []}]}, ValueError, "paper ids are not sorted"),
        ({"concepts": ["a"], "papers": [{"paper_id": "P1", "concepts": ["a", "a"]}]}, ValueError, "listed more than once"),
        ({"concepts": ["a"], "papers": [{"paper_id": "P1"}]}, TypeError, "every paper must be"),
        ({"concepts": ["a"]}, TypeError, "must be {concepts, papers}"),
    ],
)
def test_the_tools_refuse_invalid_data(tmp_path: Path, monkeypatch, data: dict, error: type, words: str) -> None:
    write_dataset(tmp_path / "small", data)
    monkeypatch.setattr(datasets, "DATA", tmp_path)
    with pytest.raises(error, match=words.replace("[", r"\[").replace("{", r"\{")):
        get_papers(build_output({"dataset": "small"}, DatasetInput), tmp_path)
    with pytest.raises(error):
        pin_venn(build_output({"dataset": "small", "a": "a", "b": "b", "c": "c"}, VennInput), tmp_path)


def test_pin_venn_echoes_the_task(tmp_path: Path) -> None:
    given = {"dataset": "standard", "a": "summarization", "b": "large language model", "c": "chain of thought"}
    pin = pin_venn(build_output(given, VennInput), tmp_path)
    assert (pin.dataset, pin.a, pin.b, pin.c) == (Dataset.STANDARD, "summarization", "large language model", "chain of thought")
    assert (pin.expected.dataset, pin.expected.a, pin.expected.b, pin.expected.c) == (pin.dataset, pin.a, pin.b, pin.c)


async def talk(workspace, script):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "plugin"],
        cwd=str(ROOT),
        env={**os.environ, "NANOAGENT_WORKSPACE": str(workspace), "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        return await script(session)


def test_the_server_offers_every_tool_and_answers_over_stdio(tmp_path: Path) -> None:
    async def script(session):
        offered = {tool.name: tool for tool in (await session.list_tools()).tools}
        assert set(offered) == {tool.value for tool in Tool}
        for tool, (given, made) in TOOLS.items():
            assert offered[tool.value].inputSchema == output_schema(given)
            assert offered[tool.value].outputSchema == output_schema(made)
        papers = await session.call_tool("get_papers", {"dataset": "small"})
        assert not papers.isError and len(papers.structuredContent["papers"]) == 8
        a, b, c = QUERIES["edge-cases"][0]
        pin = await session.call_tool("pin_venn", {"dataset": "edge-cases", "a": a, "b": b, "c": c})
        assert not pin.isError and pin.structuredContent["expected"]["regions"]["b_only"] == ["E11", "E2"]
        bad = await session.call_tool("get_papers", {"dataset": "huge"})
        assert bad.isError and "'huge' is not one of" in bad.content[0].text
        same = await session.call_tool("pin_venn", {"dataset": "small", "a": "nlp", "b": "nlp", "c": "speech"})
        assert same.isError and "a, b and c must be distinct" in same.content[0].text

    anyio.run(talk, tmp_path, script)
    assert list(tmp_path.iterdir()) == [], "the tools never write the workspace"


def test_the_verifier_and_the_server_share_one_rules_module() -> None:
    assert (ROOT / "verifier" / "rules.py").read_bytes() == (ROOT / "src" / "plugin" / "rules.py").read_bytes()


def test_the_verifier_imports_only_the_standard_library_and_the_contract() -> None:
    for path in (ROOT / "verifier").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                assert top == "contract" or top in sys.stdlib_module_names, f"{path.name} imports {name}"


def test_the_manifest_restates_every_rule_and_names_every_tool() -> None:
    manifest = (ROOT / "nanoagent.yaml").read_text(encoding="utf-8")
    for rule in ("V1.", "V2.", "V3.", "V4."):
        assert f"      {rule} " in manifest, rule
    for tool in Tool:
        assert f"  {tool.value}:\n    description:" in manifest
    assert "tools: [get_papers]" in manifest and "reference: pin_venn" in manifest
    assert "hosts: []" in manifest and "write: [workspace]" in manifest and "spec: nanoagent.plugin/v0001" in manifest
    assert '"language model" does not match "large language model"' in manifest and "never re-sorted" in manifest
    ending = (
        "Do not restate the rules. In your thinking, list only papers that have at least one of the three concepts, "
        "each with its region. Then immediately call submit with the full output.\n"
    )
    assert manifest.endswith(ending)


def test_the_readme_opens_with_the_manifest_name_and_lists_the_queries() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# GapMap Venn Regions\n")
    for dataset, triples in QUERIES.items():
        for a, b, c in triples:
            assert f"| `{dataset}` | `{a}` | `{b}` | `{c}` |" in readme, (dataset, a, b, c)


def test_ui_draws_every_output_from_its_schema() -> None:
    assert sorted(path.name for path in (ROOT / "ui").iterdir()) == ["renderers.css", "renderers.js"]
    assert (ROOT / "ui" / "renderers.js").read_text(encoding="utf-8").strip() == "export default {};"
    assert (ROOT / "ui" / "renderers.css").read_text(encoding="utf-8") == ""


def test_every_tool_is_its_own_module_taking_its_input_and_the_workspace() -> None:
    import inspect

    from plugin.__main__ import IMPLEMENTATIONS

    assert set(IMPLEMENTATIONS) == set(Tool)
    for tool, implementation in IMPLEMENTATIONS.items():
        assert implementation.__module__ == f"plugin.{tool.value}"
        parameters = list(inspect.signature(implementation).parameters.values())
        assert [parameter.name for parameter in parameters] == ["given", "workspace"]
        assert parameters[0].annotation == TOOLS[tool][0].__name__ and parameters[1].annotation == "Path"
