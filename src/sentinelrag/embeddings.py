"""Embedding providers with a dependency-free local vector fallback.

The local provider uses signed feature hashing so the app runs privately without
model downloads. Set an OpenAI-compatible key to use semantic embeddings in a
production/demo deployment.
"""
import hashlib
import math
import re
from collections import Counter

DIMENSIONS = 384
TOKEN = re.compile(r"[a-zA-Z0-9_'-]+")


def local_embed(text: str) -> list[float]:
    vector = [0.0] * DIMENSIONS
    counts = Counter(TOKEN.findall(text.lower()))
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % DIMENSIONS
        sign = 1.0 if (value >> 1) & 1 else -1.0
        vector[index] += sign * (1 + math.log(count))
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 7) for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))
