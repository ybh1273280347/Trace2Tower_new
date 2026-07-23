from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

from trace2tower.methods.trace2tower.core.models import SegmentInstance
from trace2tower.methods.trace2tower.induction.skills import HighSkillCard, MidSkillCard
from trace2tower.methods.trace2tower.artifacts.tower import TowerSnapshot


def format_mid(card: MidSkillCard) -> list[str]:
    lines = [f"Name: {card.name}", f"Description: {card.description}", "Procedure:"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(card.procedure, 1))
    lines.append("Constraints:")
    lines.extend(f"- {item}" for item in card.constraints)
    lines.append(
        "Grounding: " + ", ".join(action.value for action in card.grounding_actions)
    )
    return lines


def format_high(card: HighSkillCard) -> list[str]:
    lines = [f"Name: {card.name}", f"Description: {card.description}", "Procedure:"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(card.procedure, 1))
    return lines


def main(options: argparse.Namespace) -> int:
    old = TowerSnapshot.from_record(json.loads(options.old_tower.read_text(encoding="utf-8")))
    new = TowerSnapshot.from_record(json.loads(options.new_tower.read_text(encoding="utf-8")))
    report = json.loads(options.build_report.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in options.preprocessed.read_text(encoding="utf-8").splitlines()
        if line
    ]
    segments = {
        segment.segment_id: segment
        for record in records
        for segment in (SegmentInstance.from_record(item) for item in record["segments"])
    }
    old_mid = {card.skill_id: card for card in old.mid_cards}
    old_high = {card.skill_id: card for card in old.high_cards}
    old_paths = {path.path_id: path for path in old.high_paths}
    sources = report["projected_mid_objective_sources"]
    rendered_skill_ids = set(report["rendered_skill_ids"])

    lines = [
        f"# Refined Card Review: {new.snapshot_id}",
        "",
        f"Source tower: `{old.snapshot_id}`",
        "",
        "## Changed Mid Cards",
        "",
    ]
    for card in new.mid_cards:
        if card.skill_id not in rendered_skill_ids:
            continue
        old_card = old_mid.get(card.skill_id)
        member_segments = [segments[segment_id] for segment_id in card.member_segment_ids]
        scores = [segment.trajectory_score for segment in member_segments]
        events = Counter(
            segment.event_type.value if segment.event_type else "NONE"
            for segment in member_segments
        )
        lines.extend(
            [
                f"### {card.skill_id}",
                "",
                f"Sources: `{', '.join(sources.get(card.skill_id, ())) or 'new-only'}`",
                f"Members: {len(member_segments)}; mean score: {fmean(scores):.4f}; full-score segments: {sum(score >= 0.999 for score in scores)}",
                "Events: " + ", ".join(f"{key}={value}" for key, value in sorted(events.items())),
                "",
                "New card:",
                "",
                *format_mid(card),
                "",
            ]
        )
        for source_id in sources.get(card.skill_id, ()):
            if source_id not in old_mid:
                continue
            lines.extend(
                [
                    f"Source card `{source_id}`:",
                    "",
                    *format_mid(old_mid[source_id]),
                    "",
                ]
            )

    lines.extend(["## Changed High Cards", ""])
    for card in new.high_cards:
        if card.skill_id not in rendered_skill_ids:
            continue
        old_card = old_high.get(card.skill_id)
        new_path = next(path for path in new.high_paths if path.path_id == card.skill_id)
        old_path = old_paths.get(card.skill_id)
        lines.extend(
            [
                f"### {card.skill_id}",
                "",
                f"Children: `{', '.join(new_path.ordered_mid_ids)}`",
                f"Positive support: {new_path.positive_support:.4f}; negative support: {new_path.negative_support:.4f}; contrastive score: {new_path.contrastive_score:.4f}",
                "",
                "New card:",
                "",
                *format_high(card),
                "",
            ]
        )
        if old_card:
            lines.extend(
                [
                    f"Old children: `{', '.join(old_path.ordered_mid_ids)}`",
                    "",
                    "Old card:",
                    "",
                    *format_high(old_card),
                    "",
                ]
            )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(options.output.as_posix())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-tower", type=Path, required=True)
    parser.add_argument("--new-tower", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
