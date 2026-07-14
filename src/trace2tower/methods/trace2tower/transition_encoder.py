from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import struct
from pathlib import Path
from typing import Sequence

from trace2tower.llm_runtime import CommonLLMRuntime


class TransitionEncoder:
    def __init__(
        self,
        runtime: CommonLLMRuntime,
        *,
        cache_path: Path,
        model: str,
        dimension: int,
        batch_size: int,
    ):
        if dimension <= 0 or batch_size <= 0:
            raise ValueError("dimension and batch size must be positive")
        self.runtime = runtime
        self.cache_path = cache_path
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.cached_unique_text_count = 0
        self.embedded_unique_text_count = 0
        self.embedding_request_count = 0
        self.embedding_input_tokens = 0

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        unique_texts = dict(zip(hashes, texts))
        cached = self._read_cache(tuple(unique_texts))
        missing_hashes = [content_hash for content_hash in unique_texts if content_hash not in cached]
        self.cached_unique_text_count = len(cached)
        self.embedded_unique_text_count = len(missing_hashes)

        batches = [
            missing_hashes[index : index + self.batch_size]
            for index in range(0, len(missing_hashes), self.batch_size)
        ]
        async def embed_batch(batch: list[str]) -> None:
            result = await self.runtime.embed(
                [unique_texts[content_hash] for content_hash in batch]
            )
            if len(result.vectors) != len(batch):
                raise ValueError("embedding provider returned the wrong batch size")
            self.embedding_request_count += 1
            self.embedding_input_tokens += result.usage.input_tokens or 0
            new_vectors = {}
            for content_hash, vector in zip(batch, result.vectors):
                if len(vector) != self.dimension:
                    raise ValueError("embedding provider returned the wrong dimension")
                new_vectors[content_hash] = tuple(vector)
            self._write_cache(new_vectors)

        results = await asyncio.gather(
            *(embed_batch(batch) for batch in batches),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise errors[0]
        cached.update(self._read_cache(tuple(missing_hashes)))
        return tuple(cached[content_hash] for content_hash in hashes)

    def _read_cache(self, content_hashes: tuple[str, ...]) -> dict[str, tuple[float, ...]]:
        if not content_hashes or not self.cache_path.exists():
            return {}
        records = {}
        with sqlite3.connect(self.cache_path) as database:
            for index in range(0, len(content_hashes), 500):
                batch = content_hashes[index : index + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = database.execute(
                    f"SELECT content_hash, dimension, vector FROM embeddings "
                    f"WHERE model = ? AND content_hash IN ({placeholders})",
                    (self.model, *batch),
                )
                for content_hash, dimension, payload in rows:
                    if dimension != self.dimension:
                        raise ValueError("cached embedding dimension does not match configuration")
                    records[content_hash] = struct.unpack(f"<{dimension}f", payload)
        return records

    def _write_cache(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "model TEXT NOT NULL, content_hash TEXT NOT NULL, dimension INTEGER NOT NULL, "
                "vector BLOB NOT NULL, PRIMARY KEY(model, content_hash))"
            )
            database.executemany(
                "INSERT OR IGNORE INTO embeddings(model, content_hash, dimension, vector) "
                "VALUES (?, ?, ?, ?)",
                (
                    (
                        self.model,
                        content_hash,
                        self.dimension,
                        struct.pack(f"<{self.dimension}f", *vector),
                    )
                    for content_hash, vector in vectors.items()
                ),
            )
