from __future__ import annotations

from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.alfworld_task_adapter import AlfworldTaskAdapter
from trace2tower.methods.trace2tower.task_conditioning import DomainTaskAdapter
from trace2tower.methods.trace2tower.webshop_task_adapter import WebshopTaskAdapter


def task_adapter_for(benchmark: Benchmark) -> DomainTaskAdapter:
    if benchmark is Benchmark.ALFWORLD:
        return AlfworldTaskAdapter()
    if benchmark is Benchmark.WEBSHOP:
        return WebshopTaskAdapter()
    raise ValueError(f"no task adapter registered for benchmark: {benchmark}")

