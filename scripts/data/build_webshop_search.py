from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Datasets/webshop/lucene_docs/documents.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Datasets/webshop/search.sqlite"),
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    options = parser.parse_args()

    options.output.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(options.output)
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=NORMAL")
    database.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS products_fts "
        "USING fts5(asin UNINDEXED, contents, tokenize='porter unicode61')"
    )
    database.execute(
        "CREATE TABLE IF NOT EXISTS build_state ("
        "source_path TEXT PRIMARY KEY, source_size INTEGER NOT NULL, "
        "indexed_lines INTEGER NOT NULL)"
    )

    source_path = str(options.source.resolve())
    source_size = options.source.stat().st_size
    state = database.execute(
        "SELECT source_size, indexed_lines FROM build_state WHERE source_path = ?", (source_path,)
    ).fetchone()
    if state is None:
        indexed_lines = 0
        database.execute(
            "INSERT INTO build_state(source_path, source_size, indexed_lines) VALUES (?, ?, 0)",
            (source_path, source_size),
        )
        database.commit()
    else:
        if state[0] != source_size:
            raise RuntimeError("source size changed after indexing started")
        indexed_lines = state[1]

    batch = []
    with options.source.open(encoding="utf-8") as source_file:
        for _ in range(indexed_lines):
            next(source_file)
        for line in source_file:
            document = json.loads(line)
            batch.append((document["id"], document["contents"]))
            if len(batch) < options.batch_size:
                continue

            with database:
                database.executemany(
                    "INSERT INTO products_fts(asin, contents) VALUES (?, ?)", batch
                )
                indexed_lines += len(batch)
                database.execute(
                    "UPDATE build_state SET indexed_lines = ? WHERE source_path = ?",
                    (indexed_lines, source_path),
                )
            batch.clear()
            if indexed_lines % 100000 == 0:
                print(f"indexed {indexed_lines}", flush=True)

    if batch:
        with database:
            database.executemany("INSERT INTO products_fts(asin, contents) VALUES (?, ?)", batch)
            indexed_lines += len(batch)
            database.execute(
                "UPDATE build_state SET indexed_lines = ? WHERE source_path = ?",
                (indexed_lines, source_path),
            )
    database.execute("INSERT INTO products_fts(products_fts) VALUES ('optimize')")
    database.commit()
    database.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    database.close()
    print(f"indexed {indexed_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
