import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import FakeEngine, identity_vector, make_face

from backend.services.face_engine import normalize
from backend.services.face_recognition import EmbeddingStore

spec = importlib.util.spec_from_file_location(
    "reindex", Path(__file__).resolve().parent.parent / "scripts" / "reindex_from_photos.py")
reindex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reindex)


def test_normalize_gives_unit_vector_and_cosine_is_dot_product():
    a, b = np.array([3.0, 4.0] + [0.0] * 126), np.array([4.0, 3.0] + [0.0] * 126)
    na, nb = normalize(a), normalize(b)
    assert abs(np.linalg.norm(na) - 1) < 1e-6
    assert abs(float(na @ nb) - 24 / 25) < 1e-6
    assert na.dtype == np.float32


def test_normalize_rejects_degenerate_vectors():
    for bad in (np.zeros(128), np.full(128, np.nan)):
        with pytest.raises(ValueError):
            normalize(bad)


def test_reindex_from_saved_photos(tmp_path):
    known = tmp_path / "known" / "Athul"
    known.mkdir(parents=True)
    (known / "metadata.json").write_text(json.dumps({"name": "Athul", "details": "Student"}))
    for i in range(3):
        cv2.imwrite(str(known / f"{i}.jpg"), np.full((200, 200, 3), 90, np.uint8))
    (tmp_path / "known" / "NoMeta").mkdir()
    engine = FakeEngine()
    engine.faces = [make_face((20, 20, 100, 100), 5)]
    store = EmbeddingStore(tmp_path / "emb", legacy_dir=tmp_path / "legacy")
    rows = reindex.reindex(engine, store, tmp_path / "known")
    assert ("Athul", 3, 0, "ok") in rows and any("skipped" in r[3] for r in rows)
    assert store.people() == ["Athul"]
    assert len(list((known).glob("*.jpg"))) == 3          # photos never modified
