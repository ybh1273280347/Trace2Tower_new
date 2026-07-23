from __future__ import annotations

import numpy as np

from trace2tower.components.llm_runtime import CommonLLMRuntime


class SkillXEmbeddingAdapter:
    def __init__(self, runtime: CommonLLMRuntime):
        self.runtime = runtime
        self.input_tokens = 0

    async def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        result = await self.runtime.embed(texts)
        self.input_tokens += result.usage.input_tokens or 0
        vectors = np.asarray(result.vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("SkillX embedding response does not match the request")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
