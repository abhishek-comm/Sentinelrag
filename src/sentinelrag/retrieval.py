import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .embeddings import cosine, local_embed
from .storage import connection

TOKEN = re.compile(r"[a-zA-Z0-9_'-]+")
STOP_WORDS = {"a", "an", "and", "are", "can", "does", "for", "how", "is", "of", "the", "to",
              "what", "when", "where", "which", "who", "with", "will", "would", "many", "permitted",
              "policy", "company"}


def tokens(text: str) -> list[str]:
    raw = [token for token in TOKEN.findall(text.lower()) if token not in STOP_WORDS]
    # Lightweight normalization keeps the demo self-contained (remote/remotely,
    # report/reported) while a production deployment would use embeddings.
    normalized: list[str] = []
    for token in raw:
        if token.endswith("ly") and len(token) > 5 or token.endswith("ed") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        normalized.append(token)
    return normalized


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    document_id: str
    filename: str
    page: int
    text: str
    score: float


def _rows(document_ids: list[str] | None) -> list[dict]:
    query = """SELECT c.id, c.document_id, d.filename, c.page, c.safe_text, c.embedding_json
               FROM chunks c JOIN documents d ON d.id = c.document_id"""
    params: list[str] = []
    if document_ids:
        placeholders = ",".join("?" for _ in document_ids)
        query += f" WHERE c.document_id IN ({placeholders})"
        params = document_ids
    with connection() as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def _bm25(query_tokens: list[str], docs: list[list[str]]) -> list[float]:
    if not docs:
        return []
    df = Counter(token for doc in docs for token in set(doc))
    n = len(docs)
    avg_len = sum(map(len, docs)) / n or 1
    scores = []
    for doc in docs:
        freqs = Counter(doc)
        score = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (freqs[term] * 2.0) / (freqs[term] + 1.2 * (1 - 0.75 + 0.75 * len(doc) / avg_len))
        scores.append(score)
    return scores


def _overlap(query_tokens: list[str], docs: list[list[str]]) -> list[float]:
    query_set = set(query_tokens)
    return [len(query_set & set(doc)) / max(len(query_set), 1) for doc in docs]


def retrieve(question: str, document_ids: list[str] | None = None, limit: int = 8) -> list[RetrievedChunk]:
    rows = _rows(document_ids)
    q = tokens(question)
    docs = [tokens(row["safe_text"]) for row in rows]
    bm25 = _bm25(q, docs)
    overlap = _overlap(q, docs)
    query_vector = local_embed(question)
    vector_scores = [cosine(query_vector, json.loads(row["embedding_json"])) if row["embedding_json"] else 0.0
                     for row in rows]
    rankings: list[list[int]] = [
        sorted(range(len(rows)), key=lambda i: bm25[i], reverse=True),
        sorted(range(len(rows)), key=lambda i: overlap[i], reverse=True),
        sorted(range(len(rows)), key=lambda i: vector_scores[i], reverse=True),
    ]
    rrf: defaultdict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            rrf[index] += 1 / (60 + rank)
    ordered = sorted(rrf, key=rrf.get, reverse=True)
    retrieved: list[RetrievedChunk] = []
    seen_text: set[str] = set()
    for i in ordered:
        content_key = " ".join(rows[i]["safe_text"].lower().split())
        relevance = 0.5 * overlap[i] + 0.25 * min(bm25[i] / 5, 1.0) + 0.25 * max(vector_scores[i], 0.0)
        if content_key in seen_text or relevance <= 0.05:
            continue
        seen_text.add(content_key)
        retrieved.append(RetrievedChunk(
            id=rows[i]["id"], document_id=rows[i]["document_id"], filename=rows[i]["filename"],
            page=rows[i]["page"], text=rows[i]["safe_text"], score=round(relevance, 4),
        ))
        if len(retrieved) >= limit:
            break
    return retrieved
