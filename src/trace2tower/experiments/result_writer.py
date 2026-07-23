from __future__ import annotations

from trace2tower.core.manifests import ManifestEntry
from trace2tower.core.results import EpisodeKey, EpisodeResult, MethodName
from trace2tower.experiments.checkpoint import EpisodeCheckpoint


class EpisodeResultWriter:
    def __init__(self, checkpoint: EpisodeCheckpoint):
        self.checkpoint = checkpoint

    def is_completed(self, entry: ManifestEntry, method: MethodName) -> bool:
        return self.checkpoint.is_completed(
            EpisodeKey(
                benchmark=entry.benchmark,
                split=entry.split,
                method=method,
                sample_id=entry.sample_id,
                repeat_id=entry.repeat_id,
            )
        )

    def write(self, result: EpisodeResult) -> bool:
        return self.checkpoint.write_result(result.to_record())

    def write_error(self, entry: ManifestEntry, method: MethodName, error: str) -> None:
        self.checkpoint.write_error(
            EpisodeKey(
                benchmark=entry.benchmark,
                split=entry.split,
                method=method,
                sample_id=entry.sample_id,
                repeat_id=entry.repeat_id,
            ),
            error,
        )
