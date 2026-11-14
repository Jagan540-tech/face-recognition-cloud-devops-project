"""Runs only when the real ONNX files are present (python scripts/download_models.py).
These check that the OpenCV wiring works with the real YuNet/SFace networks. They do NOT measure
recognition accuracy - that needs real faces (see README, 'Calibrating the threshold')."""
import numpy as np
import pytest

import config

pytestmark = pytest.mark.skipif(
    not (config.DETECTOR_MODEL.exists() and config.RECOGNIZER_MODEL.exists()),
    reason="real model files not downloaded")


@pytest.fixture(scope="module")
def engine():
    from backend.services.face_engine import FaceEngine
    return FaceEngine()


def test_real_detector_on_blank_and_noise_frames(engine):
    assert engine.detect(np.full((480, 640, 3), 128, np.uint8)) == []
    noise = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    for f in engine.detect(noise):                      # if anything fires, output must be sane
        x, y, w, h = f.box
        assert 0 <= x and 0 <= y and x + w <= 640 and y + h <= 480 and 0 <= f.score <= 1


def test_real_detector_handles_changing_frame_sizes(engine):
    for shape in ((480, 640, 3), (600, 800, 3), (240, 320, 3)):
        engine.detect(np.zeros(shape, np.uint8))


def test_real_embedding_is_unit_128d(engine):
    from backend.services.face_engine import Face
    frame = np.random.default_rng(1).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    raw = np.array([100, 100, 100, 100, 130, 130, 170, 130, 150, 160, 135, 185, 165, 185, 0.99], np.float32)
    face = Face((100, 100, 100, 100), 0.99, raw[4:14].reshape(5, 2), raw)
    v = engine.embed(frame, face)
    assert v.shape == (128,) and abs(np.linalg.norm(v) - 1) < 1e-4
