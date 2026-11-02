"""get_papers: a dataset's vocabulary and papers for the agent, exactly as stored. One tool, not two: each
call costs about 15 s of NanoAgent sandbox overhead."""

from __future__ import annotations

from pathlib import Path

from contract import DatasetInput, PapersOutput
from plugin.datasets import load, paper_list


def get_papers(given: DatasetInput, workspace: Path) -> PapersOutput:
    """The workspace is unused: the data is bundled, and the tool writes nothing."""
    concepts, papers = load(given.dataset)
    return PapersOutput(concepts=list(concepts), papers=paper_list(papers))
