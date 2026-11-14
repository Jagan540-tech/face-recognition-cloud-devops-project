import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from app import create_app  # noqa: E402
from backend.services.face_engine import Face  # noqa: E402
from backend.services.face_recognition import EmbeddingStore  # noqa: E402

MODEL_ID = config.EMBEDDING_MODEL_ID


def make_face(box, identity):
    """A synthetic detection. `identity` selects the embedding the fake model will produce."""
    return Face(box=tuple(box), score=0.99, landmarks=np.zeros((5, 2), np.float32),
                raw=np.array([identity], dtype=np.float32))


def identity_vector(identity, noise=0.0):
    v = np.random.default_rng(int(identity)).normal(size=128)
    if noise:
        v = v + np.random.default_rng(999).normal(size=128) * noise
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeEngine:
    """Stands in for FaceEngine: the test decides which faces 'appear' in the next frame.
    Different identities give (almost) orthogonal vectors, the same identity an identical one.
    This proves the application logic, NOT real-world recognition accuracy."""
    model_id = MODEL_ID
    dim = 128

    def __init__(self):
        self.faces, self.last_shape = [], None

    def detect(self, frame):
        self.last_shape = frame.shape
        return list(self.faces)

    def embed(self, frame, face):
        return identity_vector(face.raw[0])


def jpeg_data_url(size=(480, 640), color=120):
    img = np.full((*size, 3), color, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SAMPLES_PER_PERSON", 3)
    engine = FakeEngine()
    store = EmbeddingStore(tmp_path / "embeddings", legacy_dir=tmp_path / "legacy")
    app = create_app(engine=engine, store=store, known_dir=tmp_path / "known")
    return app.test_client(), engine, store, tmp_path


def register(client, engine, name, role, identity, box=(100, 100, 120, 140)):
    engine.faces = [make_face(box, identity)]
    for _ in range(3):
        client.post("/api/register", json={"session_id": name, "name": name, "role": role,
                                           "image": jpeg_data_url()})
    r = client.post("/api/register", json={"session_id": name, "name": name, "role": role,
                                           "finish": True})
    assert r.status_code == 200, r.get_json()
