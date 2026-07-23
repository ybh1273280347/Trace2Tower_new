from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.methods.trace2tower.artifacts.tower import TowerSnapshot


def main(options: argparse.Namespace) -> int:
    gate = json.loads(options.gate_report.read_text(encoding="utf-8"))
    if gate["protocol_id"] != "alfworld-deployment-optimization-v1-gate":
        raise ValueError("selection requires the frozen ALFWorld deployment gate report")
    if gate["selected"] is not None:
        raise ValueError("this selection command is only for a no-op gate result")
    if any(candidate["gate_pass"] for candidate in gate["candidates"].values()):
        raise ValueError("no-op cannot be selected while a candidate passes the gate")
    tower = TowerSnapshot.from_record(json.loads(options.base_tower.read_text(encoding="utf-8")))
    content = {
        "protocol_id": "alfworld-deployment-optimization-v1-selection",
        "selected_candidate": "v0_noop",
        "snapshot_id": tower.snapshot_id,
        "tower_path": options.base_tower.as_posix(),
        "tower_sha256": hashlib.sha256(options.base_tower.read_bytes()).hexdigest(),
        "deployment_policy": None,
        "gate_report_path": options.gate_report.as_posix(),
        "gate_report_sha256": hashlib.sha256(options.gate_report.read_bytes()).hexdigest(),
        "reason": "all modified candidates failed the frozen noninferiority gate",
    }
    selection_payload = json.dumps(content, sort_keys=True, separators=(",", ":"))
    payload = {
        "selection_id": f"selection_{hashlib.sha256(selection_payload.encode()).hexdigest()[:16]}",
        **content,
    }
    write_json(options.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--base-tower", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
