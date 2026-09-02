"""Run a local, repeatable SentinelRAG benchmark against an ingested corpus."""
import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinelrag.answering import answer
from sentinelrag.ingestion import ingest
from sentinelrag.storage import initialize


async def run(dataset: Path, document: Path | None) -> dict:
    initialize()
    if document:
        ingest(document, document.name)
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for case in cases:
        response = await answer(case["question"])
        pages = {citation["page"] for citation in response["citations"]}
        expected = set(case["expected_pages"])
        retrieved_expected = bool(pages & expected) if expected else response["status"] == "abstained"
        answer_text = response["answer"].lower()
        keyword_pass = all(word.lower() in answer_text for word in case["keywords"])
        abstention_pass = (response["status"] == "abstained") == case["expected_abstain"]
        results.append({"id": case["id"], "status": response["status"], "latency_ms": response["latency_ms"],
                        "retrieval_pass": retrieved_expected, "keyword_pass": keyword_pass,
                        "abstention_pass": abstention_pass})
    total = max(len(results), 1)
    report = {
        "created_at": datetime.now(UTC).isoformat(), "cases": total,
        "retrieval_recall": round(sum(x["retrieval_pass"] for x in results) / total, 3),
        "grounded_answer_keyword_rate": round(sum(x["keyword_pass"] for x in results) / total, 3),
        "abstention_accuracy": round(sum(x["abstention_pass"] for x in results) / total, 3),
        "mean_latency_ms": round(sum(x["latency_ms"] for x in results) / total, 1), "details": results,
    }
    output = ROOT / "data/evaluation/results" / f"report-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved {output.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ingest", type=Path, help="Optional document to ingest before the run")
    arguments = parser.parse_args()
    asyncio.run(run(arguments.dataset, arguments.ingest))
