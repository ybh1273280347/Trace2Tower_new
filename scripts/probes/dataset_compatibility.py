from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as parquet

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    alfworld_manifest = json.loads(
        (ROOT / "third_party/AgentBench/data/alfworld/train_valid.json").read_text(encoding="utf-8")
    )
    manifest_paths = [path for group in alfworld_manifest.values() for path in group]
    games_root = ROOT / "Datasets/alfworld/games/games"
    resolved_agentbench_games = 0
    for value in manifest_paths:
        relative = Path(value)
        if relative.parts[0] == "json_2.1.1":
            relative = Path(*relative.parts[1:])
        resolved_agentbench_games += (games_root / relative).is_file()

    alfworld_splits = {}
    all_local_games_resolved = True
    for split in ("train", "valid_seen", "valid_unseen"):
        table = parquet.read_table(ROOT / f"Datasets/alfworld/{split}.parquet")
        extra_info = table.column("extra_info").combine_chunks()
        relative_paths = extra_info.field("game_relative_path").to_pylist()
        resolved = sum((games_root / relative_path).is_file() for relative_path in relative_paths)
        alfworld_splits[split] = {"rows": table.num_rows, "resolved_games": resolved}
        all_local_games_resolved &= resolved == table.num_rows

    goals = json.loads((ROOT / "Datasets/webshop/goals.json").read_text(encoding="utf-8"))
    database_path = ROOT / "Datasets/webshop/products.sqlite"
    with sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True) as database:
        product_count = database.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        first_product = database.execute(
            "SELECT product_json FROM products WHERE asin = ?", (goals[0]["asin"],)
        ).fetchone()
    product = json.loads(first_product[0]) if first_product else {}
    reward_fields = {
        "Title",
        "Description",
        "BulletPoints",
        "Attributes",
        "query",
        "product_category",
        "options",
        "_price",
    }
    webshop_splits = {
        split: parquet.read_metadata(ROOT / f"Datasets/webshop/{split}.parquet").num_rows
        for split in ("train", "test")
    }

    result = {
        "alfworld": {
            "local_splits": alfworld_splits,
            "all_local_games_resolved": all_local_games_resolved,
            "agentbench_train_count": len(manifest_paths),
            "agentbench_train_overlap": resolved_agentbench_games,
        },
        "webshop": {
            "local_splits": webshop_splits,
            "goal_count": len(goals),
            "product_count": product_count,
            "first_goal_product_found": first_product is not None,
            "reward_fields_present": reward_fields <= product.keys(),
            "lucene_index_present": (ROOT / "Datasets/webshop/lucene_index/segments_1").is_file(),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    webshop_ready = all(
        (
            result["webshop"]["first_goal_product_found"],
            result["webshop"]["reward_fields_present"],
            result["webshop"]["lucene_index_present"],
        )
    )
    return 0 if all_local_games_resolved and webshop_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
