#!/usr/bin/env python3
"""Rebuild the embedding store from saved registration photos (data/known_faces/<Name>/*.jpg
+ metadata.json). Use it after changing the recognition model, instead of re-registering,
provided the photos still exist. Photos with no face or several faces are skipped and reported.

    python scripts/reindex_from_photos.py            # writes data/embeddings/embeddings.npz
    python scripts/reindex_from_photos.py --dry-run  # only report what would happen
Any existing store is backed up first by the store itself; photos are never modified.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2  # noqa: E402

import config  # noqa: E402
from backend.services.face_engine import FaceEngine  # noqa: E402
from backend.services.face_recognition import EmbeddingStore  # noqa: E402


def reindex(engine, store, known_dir, dry_run=False):
    summary = []
    for folder in sorted(p for p in Path(known_dir).iterdir() if p.is_dir()):
        meta_file = folder / "metadata.json"
        if not meta_file.exists():
            summary.append((folder.name, 0, 0, "no metadata.json - skipped"))
            continue
        meta = json.loads(meta_file.read_text())
        vectors, skipped = [], 0
        for img_path in sorted(folder.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            faces = engine.detect(img) if img is not None else []
            if len(faces) != 1:
                skipped += 1
                continue
            vectors.append(engine.embed(img, faces[0]))
        if vectors and not dry_run:
            store.add(vectors, meta["name"], meta.get("details", ""))
        summary.append((meta["name"], len(vectors), skipped, "ok" if vectors else "no usable photos"))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = reindex(FaceEngine(), EmbeddingStore(), config.KNOWN_FACES_DIR, args.dry_run)
    for name, ok, skipped, note in rows:
        print(f"{name}: {ok} embeddings, {skipped} photos skipped ({note})")
    if not rows:
        print("No registered people found in", config.KNOWN_FACES_DIR)


if __name__ == "__main__":
    main()
