# SentinelRAG

**A citation-first, self-evaluating document intelligence agent.** SentinelRAG answers questions over uploaded documents only when it can retrieve sufficient evidence. Every answer is tied to document and page citations, and every response is recorded for audit and evaluation.

## Why this is not another PDF chatbot

SentinelRAG treats retrieval and answer quality as measurable engineering problems:

- hybrid BM25 + vector retrieval with reciprocal-rank fusion;
- page-aware ingestion for PDFs, DOCX, and text documents;
- confidence-aware abstention when evidence is weak;
- source citations with supporting excerpts;
- prompt-injection filtering for hostile instructions embedded in documents;
- a JSONL evaluation harness for retrieval recall, citation correctness, abstention, latency, and answer grounding;
- trace records and user feedback for investigating failures.

## Architecture

```text
Upload -> Parser/OCR-ready ingestion -> Page-aware chunks -> SQLite document store
                                                       |-> BM25 search
Question -> query guard -> hybrid retriever -> RRF fusion -> evidence gate
                                                       -> answer provider -> citations + trace
Evaluation JSONL ---------------------------------------------------------> metrics report
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env  # Windows (or cp .env.example .env)
uvicorn sentinelrag.main:app --reload
```

Open `http://127.0.0.1:8000/docs`. The system runs without an LLM key using an extractive answer fallback. To use OpenAI-compatible models, set the three `SENTINEL_LLM_*` variables in `.env`.

For the built-in dashboard, open `http://127.0.0.1:8000/`.

## Demo

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload -F "file=@data/demo/company_handbook.txt"
curl -X POST http://127.0.0.1:8000/api/v1/questions -H "Content-Type: application/json" -d "{\"question\":\"How many remote days can employees work?\"}"
python scripts/run_evaluation.py --dataset data/evaluation/demo_cases.jsonl
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/documents/upload` | Ingest PDF, DOCX, TXT, or Markdown |
| `GET` | `/api/v1/documents` | List documents |
| `POST` | `/api/v1/questions` | Ask a cited question |
| `POST` | `/api/v1/feedback` | Record answer feedback |
| `GET` | `/api/v1/traces` | Inspect recent answer traces |
| `GET` | `/health` | Readiness probe |

## Evaluation

The evaluation runner uses expected document/page citations and expected abstention behavior. It produces a timestamped report in `data/evaluation/results/`.

```bash
python scripts/run_evaluation.py --dataset data/evaluation/demo_cases.jsonl
pytest -q
```

To run entirely in Docker:

```bash
cp .env.example .env
docker compose up --build
```

## Interview talking points

1. **Why hybrid retrieval?** BM25 captures exact policy/product names while vector retrieval supplies a second similarity signal. RRF blends them without score calibration.
2. **How is hallucination reduced?** Answers are conditioned on retrieved evidence, citations are mandatory, and a confidence gate abstains before generation.
3. **What is evaluated?** Retrieval, citations, groundedness proxy, abstention correctness, latency, and feedback—not just fluent answers.
4. **Production next steps:** replace SQLite with Postgres/pgvector, add a cross-encoder reranker, async ingestion workers, OpenTelemetry exporter, authentication, and dataset/version registries.

## Limitations

The local vector provider is a private, dependency-free feature-hashing baseline. For strong semantic retrieval, replace it with a hosted embedding model or local sentence-transformer and persist returned vectors. Production use should also adopt human-validated evaluation labels. OCR requires `pip install -e ".[ocr]"` plus a system Tesseract installation; image-only PDFs are reported clearly when it is unavailable.
