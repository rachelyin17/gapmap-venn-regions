"""Each tool's and task's typed input and output (see nanoagent.plugin.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    PIN_VENN = "pin_venn"
    GET_PAPERS = "get_papers"


class Task(Enum):
    VENN_REGIONS = "venn-regions"


class Dataset(Enum):
    SMALL = "small"
    STANDARD = "standard"
    EDGE_CASES = "edge-cases"


DATASET = "Which bundled dataset: small (simple cases), standard (realistic NLP concepts) or edge-cases (every boundary case)"
CONCEPT = "A concept: lowercase words of a-z and 0-9 joined by single spaces"


@dataclass
class DatasetInput(Output):
    dataset: Annotated[Dataset, DATASET]


@dataclass
class VennInput(Output):
    dataset: Annotated[Dataset, DATASET]
    a: Annotated[str, "Concept A: a vocabulary concept, distinct from b and c"]
    b: Annotated[str, "Concept B: a vocabulary concept, distinct from a and c"]
    c: Annotated[str, "Concept C: a vocabulary concept, distinct from a and b"]


@dataclass
class Paper(Output):
    paper_id: Annotated[str, "The paper's id, unique within the dataset"]
    concepts: Annotated[list[Annotated[str, CONCEPT]], "The paper's concepts, vocabulary concepts only, unique and sorted ascending; may be empty"]


@dataclass
class PapersOutput(Output):
    concepts: Annotated[list[Annotated[str, CONCEPT]], "The dataset's vocabulary, unique and sorted ascending; a concept may appear in no paper"]
    papers: Annotated[list[Paper], "Every paper of the dataset, sorted by paper_id"]


@dataclass
class Regions(Output):
    a_only: Annotated[list[str], "Papers with a but neither b nor c, unique and sorted ascending in Python's default string order (P10 before P2)"]
    b_only: Annotated[list[str], "Papers with b but neither a nor c, unique and sorted ascending in Python's default string order (P10 before P2)"]
    c_only: Annotated[list[str], "Papers with c but neither a nor b, unique and sorted ascending in Python's default string order (P10 before P2)"]
    ab_only: Annotated[list[str], "Papers with a and b but not c, unique and sorted ascending in Python's default string order (P10 before P2)"]
    ac_only: Annotated[list[str], "Papers with a and c but not b, unique and sorted ascending in Python's default string order (P10 before P2)"]
    bc_only: Annotated[list[str], "Papers with b and c but not a, unique and sorted ascending in Python's default string order (P10 before P2)"]
    abc: Annotated[list[str], "Papers with all three, unique and sorted ascending in Python's default string order (P10 before P2)"]


@dataclass
class VennOutput(Output):
    dataset: Annotated[Dataset, "The task's dataset, copied unchanged"]
    a: Annotated[str, "The task's a, copied unchanged"]
    b: Annotated[str, "The task's b, copied unchanged"]
    c: Annotated[str, "The task's c, copied unchanged"]
    regions: Annotated[Regions, "The 7 exclusive regions of the Venn diagram of a, b and c, as given"]


@dataclass
class PinOutput(Output):
    dataset: Dataset
    a: str
    b: str
    c: str
    concepts: Annotated[list[str], "The dataset's vocabulary, sorted ascending"]
    papers: Annotated[list[Paper], "The dataset's papers, sorted by paper_id"]
    expected: Annotated[VennOutput, "The output the rules give on these papers for a, b and c"]


TOOLS = {
    Tool.PIN_VENN: (VennInput, PinOutput),
    Tool.GET_PAPERS: (DatasetInput, PapersOutput),
}
TASKS = {
    Task.VENN_REGIONS: (VennInput, VennOutput),
}
