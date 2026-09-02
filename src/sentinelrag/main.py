import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .answering import answer
from .config import settings
from .ingestion import ingest
from .schemas import AskRequest, AskResponse, DocumentSummary, FeedbackRequest
from .storage import connection, initialize, utcnow


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize()
    yield


app = FastAPI(title="SentinelRAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/documents/upload", response_model=DocumentSummary, status_code=201)
def upload(file: UploadFile = File(...)) -> dict:  # noqa: B008  # FastAPI dependency declaration
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(415, "Supported types are PDF, DOCX, TXT, and Markdown.")
    safe_name = f"{uuid.uuid4()}{suffix}"
    destination = settings.upload_dir / safe_name
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    try:
        return ingest(destination, file.filename or safe_name)
    except FileExistsError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/v1/documents", response_model=list[DocumentSummary])
def documents() -> list[dict]:
    with connection() as conn:
        rows = conn.execute("""SELECT d.id, d.filename, d.page_count, d.created_at, COUNT(c.id) AS chunk_count
                               FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
                               GROUP BY d.id ORDER BY d.created_at DESC""").fetchall()
    return [dict(row) for row in rows]


@app.post("/api/v1/questions", response_model=AskResponse)
async def ask(payload: AskRequest) -> dict:
    return await answer(payload.question, payload.document_ids)


@app.post("/api/v1/feedback", status_code=201)
def feedback(payload: FeedbackRequest) -> dict:
    with connection() as conn:
        if not conn.execute("SELECT 1 FROM traces WHERE id = ?", (payload.trace_id,)).fetchone():
            raise HTTPException(404, "Trace not found.")
        conn.execute("INSERT INTO feedback (trace_id, rating, comment, created_at) VALUES (?, ?, ?, ?)",
                     (payload.trace_id, payload.rating, payload.comment, utcnow()))
    return {"status": "recorded"}


@app.get("/api/v1/traces")
def traces(limit: int = 25) -> list[dict]:
    with connection() as conn:
        rows = conn.execute("SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (min(limit, 100),)).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/v1/metrics")
def metrics() -> dict:
    with connection() as conn:
        documents_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        traces_count = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        feedback_rows = conn.execute("SELECT rating, COUNT(*) AS count FROM feedback GROUP BY rating").fetchall()
    return {"documents": documents_count, "traces": traces_count,
            "feedback": {row["rating"]: row["count"] for row in feedback_rows}}
