from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.experiments.data.prepare_alfworld_protocol import FAMILIES
from scripts.experiments.run.rollout_no_skill_train import write_json


def family_mid_id(family: str, skill_id: str) -> str:
    return f"mid_{family}_{skill_id.removeprefix('mid_')}"


def family_high_id(family: str, skill_id: str) -> str:
    return f"high_{family}_{skill_id.removeprefix('high_')}"


def main(options: argparse.Namespace) -> int:
    clusters = []
    paths = []
    mid_cards = []
    high_cards = []
    usage = []
    family_report = {}
    for family in FAMILIES:
        root = options.family_root / family
        cluster_payload = json.loads((root / "graph" / "clusters.json").read_text(encoding="utf-8"))
        skill_root = root / options.skills_dir_name
        path_payload = json.loads(
            (skill_root / "high-paths.json").read_text(encoding="utf-8")
        )
        card_payload = json.loads(
            (skill_root / "rendered-cards.json").read_text(encoding="utf-8")
        )
        mid_ids = {
            item["cluster_id"]: family_mid_id(family, item["cluster_id"])
            for item in cluster_payload["clusters"]
        }
        high_ids = {
            item["path_id"]: family_high_id(family, item["path_id"])
            for item in path_payload["paths"]
        }
        for item in cluster_payload["clusters"]:
            clusters.append({**item, "cluster_id": mid_ids[item["cluster_id"]]})
        for item in path_payload["paths"]:
            paths.append(
                {
                    **item,
                    "path_id": high_ids[item["path_id"]],
                    "ordered_mid_ids": [mid_ids[mid_id] for mid_id in item["ordered_mid_ids"]],
                }
            )
        for item in card_payload["mid_cards"]:
            mid_cards.append({**item, "skill_id": mid_ids[item["skill_id"]]})
        for item in card_payload["high_cards"]:
            high_cards.append(
                {
                    **item,
                    "skill_id": high_ids[item["skill_id"]],
                    "ordered_mid_ids": [mid_ids[mid_id] for mid_id in item["ordered_mid_ids"]],
                }
            )
        for item in card_payload.get("usage", ()):
            skill_id = item["skill_id"]
            mapped_id = mid_ids.get(skill_id, high_ids.get(skill_id))
            usage.append({**item, "skill_id": mapped_id})
        family_report[family] = {
            "mid_count": len(mid_ids),
            "high_count": len(high_ids),
        }

    options.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(options.output_dir / "clusters.json", {"clusters": clusters})
    write_json(options.output_dir / "high-paths.json", {"paths": paths})
    write_json(
        options.output_dir / "rendered-cards.json",
        {"mid_cards": mid_cards, "high_cards": high_cards, "usage": usage},
    )
    report = {
        "family_root": options.family_root.as_posix(),
        "mid_count": len(mid_cards),
        "high_count": len(high_cards),
        "families": family_report,
    }
    write_json(options.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skills-dir-name", default="skills-final2")
    raise SystemExit(main(parser.parse_args()))
