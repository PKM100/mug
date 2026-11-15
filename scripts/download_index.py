#!/usr/bin/env python3
"""Download the MedRAG textbook corpus and build the dense index MUG retrieves from.

    cd backend && .venv/bin/python ../scripts/download_index.py            # -> ../data/index
    .venv/bin/python ../scripts/download_index.py --data-dir /data         # e.g. a Docker volume

Fetches the 18 chunked textbooks (~230 MB) from the Hugging Face dataset ``MedRAG/textbooks``
(Xiong et al., 2024) and embeds all 125,847 snippets with all-MiniLM-L6-v2 (fastembed's ONNX build),
writing ``<data-dir>/index/emb.npy`` (L2-normalised float32) and ``<data-dir>/index/meta.jsonl``.
Embedding takes roughly 10-30 minutes on a laptop CPU. Re-running reuses downloaded files.
The corpus is not redistributed here; check the dataset card for its terms of use.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

REPO = "MedRAG/textbooks"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/chunk/"
BOOKS = [
    "Anatomy_Gray", "Biochemistry_Lippincott", "Cell_Biology_Alberts", "First_Aid_Step1", "First_Aid_Step2",
    "Gynecology_Novak", "Histology_Ross", "Immunology_Janeway", "InternalMed_Harrison", "Neurology_Adams",
    "Obstentrics_Williams", "Pathology_Robbins", "Pathoma_Husain", "Pediatrics_Nelson", "Pharmacology_Katzung",
    "Physiology_Levy", "Psichiatry_DSM-5", "Surgery_Schwartz",
]  # spellings follow the dataset's file names
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def download(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0, connect=15.0)) as http:
        for i, book in enumerate(BOOKS, 1):
            out = dest / f"{book}.jsonl"
            if out.exists() and out.stat().st_size > 0:
                print(f"[{i}/{len(BOOKS)}] {book}: cached")
                continue
            tmp = out.with_suffix(".part")
            with http.stream("GET", BASE + out.name) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            tmp.rename(out)
            print(f"[{i}/{len(BOOKS)}] {book}: {out.stat().st_size / 1e6:.1f} MB")


def load(src: Path, limit: int | None) -> list[dict]:
    snippets: list[dict] = []
    for f in sorted(src.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    snippets.append({"id": d["id"], "title": d["title"],
                                     "content": d.get("content", d.get("contents", ""))})
                    if limit and len(snippets) >= limit:
                        return snippets
    return snippets


def build(snippets: list[dict], out: Path, batch: int) -> None:
    from fastembed import TextEmbedding

    out.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(EMBED_MODEL)
    texts = [s["content"] for s in snippets]
    emb = np.zeros((len(texts), 384), dtype=np.float32)
    t0 = time.time()
    for i, v in enumerate(model.embed(texts, batch_size=batch)):
        emb[i] = v
        if (i + 1) % 10000 == 0:
            print(f"  embedded {i + 1}/{len(texts)} ({time.time() - t0:.0f}s)")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    np.save(out / "emb.npy", emb)
    with open(out / "meta.jsonl", "w") as fh:
        for s in snippets:
            fh.write(json.dumps(s) + "\n")
    print(f"wrote {out}/emb.npy {emb.shape} and meta.jsonl in {time.time() - t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data",
                    help="where textbooks/ and index/ go (default: <repo>/data; the app reads MUG_DATA_DIR)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="index only the first N snippets (for testing)")
    ap.add_argument("--skip-download", action="store_true", help="use textbooks already in <data-dir>/textbooks")
    a = ap.parse_args()
    if not a.skip_download:
        download(a.data_dir / "textbooks")
    snippets = load(a.data_dir / "textbooks", a.limit)
    if not snippets:
        print("no snippets found", file=sys.stderr)
        return 1
    print(f"embedding {len(snippets)} snippets with {EMBED_MODEL}")
    build(snippets, a.data_dir / "index", a.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
