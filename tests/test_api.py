import asyncio
from pathlib import Path

import pytest

from sentinelrag.answering import answer
from sentinelrag.ingestion import ingest
from sentinelrag.storage import initialize


def test_answer_is_cited_and_unknown_question_abstains(tmp_path: Path) -> None:
    initialize()
    sample = tmp_path / "policy.txt"
    sample.write_text("Employees can work remotely three days weekly with manager approval.")
    ingest(sample, "policy.txt")
    with pytest.raises(FileExistsError):
        ingest(sample, "duplicate-policy.txt")
    known = asyncio.run(answer("How many remote days are permitted?"))
    assert known["status"] == "answered"
    assert known["citations"]
    unknown = asyncio.run(answer("What is the parental leave duration?"))
    assert unknown["status"] == "abstained"
