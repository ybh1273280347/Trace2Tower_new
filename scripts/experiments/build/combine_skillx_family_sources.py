from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.experiments.data.prepare_alfworld_protocol import FAMILIES
from scripts.experiments.data.split_alfworld_evidence_by_family import (
    FAMILY_DESCRIPTIONS,
)
from scripts.experiments.run.rollout_no_skill_train import write_json


def main(options: argparse.Namespace) -> int:
    sources = {
        family: json.loads(
            (options.family_root / family / "library.json").read_text(encoding="utf-8")
        )
        for family in FAMILIES
    }
    if any(source.get("benchmark") != "alfworld" for source in sources.values()):
        raise ValueError("SkillX family sources must all target ALFWorld")

    planning = {}
    functional = []
    atomic = []
    family_counts = {}
    for family in FAMILIES:
        label = FAMILY_DESCRIPTIONS[family]
        skills = sources[family]["skills"]
        for task, record in skills.get("planning", {}).items():
            tagged_task = f"Task family: {label}. Goal: {task}"
            planning[tagged_task] = {**record, "task": tagged_task}
        for skill_type, target in (("functional", functional), ("atomic", atomic)):
            for record in skills.get(skill_type, []):
                target.append(
                    {
                        **record,
                        "name": f"{label}: {record['name']}",
                        "document": f"Task family: {label}.\n{record['document']}",
                    }
                )
        family_counts[family] = {
            "planning": len(skills.get("planning", {})),
            "functional": len(skills.get("functional", [])),
            "atomic": len(skills.get("atomic", [])),
        }

    first = sources[FAMILIES[0]]
    combined = {
        "version": first.get("version", "1.0"),
        "benchmark": "alfworld",
        "created_at": first.get("created_at"),
        "skills": {
            "planning": planning,
            "functional": functional,
            "atomic": atomic,
        },
        "embeddings": first.get("embeddings", {}),
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(options.output, combined)
    report = {
        "output": options.output.as_posix(),
        "planning_count": len(planning),
        "functional_count": len(functional),
        "atomic_count": len(atomic),
        "families": family_counts,
    }
    write_json(options.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
