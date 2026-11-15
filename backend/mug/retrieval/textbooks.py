"""Dense retrieval over the MedRAG textbook corpus used in the MUG paper.

Reuses the paper's prebuilt index (all-MiniLM-L6-v2, 125,847 x 384, L2-normalized);
queries are embedded with fastembed's ONNX build of the same model (verified cos=1.0).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextbookIndex:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.ids: list[str] = []
        self.titles: list[str] = []
        self.contents: list[str] = []
        self.emb: np.ndarray | None = None
        self._embedder = None
        self._lock = threading.Lock()
        self.ready = False
        self.error: str | None = None

    def load(self) -> "TextbookIndex":
        with self._lock:
            if self.ready:
                return self
            try:
                from fastembed import TextEmbedding

                self.emb = np.load(self.index_dir / "emb.npy")
                with open(self.index_dir / "meta.jsonl") as fh:
                    for line in fh:
                        d = json.loads(line)
                        self.ids.append(d["id"])
                        self.titles.append(d["title"])
                        self.contents.append(d["content"])
                self._embedder = TextEmbedding(EMBED_MODEL)
                self.ready = True
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
        return self

    def embed(self, text: str) -> np.ndarray:
        v = np.asarray(next(iter(self._embedder.embed([text]))), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def search(self, query: str, k: int) -> list[dict]:
        if not self.ready or k <= 0:
            return []
        sims = self.emb @ self.embed(query)
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [
            {
                "id": self.ids[i],
                "source": "textbook",
                "title": self.titles[i].replace("_", " "),
                "content": self.contents[i],
                "url": None,
                "score": round(float(sims[i]), 4),
            }
            for i in idx
        ]
