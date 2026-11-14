import hashlib
import importlib.util
from pathlib import Path

import pytest

import config

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "download_models.py"
spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)


def spec_for(data, name="m.onnx", url=""):
    return {"file": name, "url": url, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def test_pinned_models_are_official_opencv_zoo_with_hashes():
    assert len(dm.MODELS) == 2
    for m in dm.MODELS:
        assert m["url"].startswith("https://github.com/opencv/opencv_zoo/")
        assert len(m["sha256"]) == 64 and m["size"] > 0
    files = {m["file"] for m in dm.MODELS}
    assert config.DETECTOR_MODEL.name in files and config.RECOGNIZER_MODEL.name in files


def test_check_accepts_good_rejects_bad(tmp_path):
    good = b"x" * 100
    s = spec_for(good)
    f = tmp_path / "m.onnx"
    assert dm.check(f, s) == "missing"
    f.write_bytes(good)
    assert dm.check(f, s) is None
    f.write_bytes(good[:50])
    assert "wrong size" in dm.check(f, s)
    f.write_bytes(b"y" * 100)
    assert "wrong SHA-256" in dm.check(f, s)


def test_download_verifies_and_installs_atomically(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"model-bytes" * 10)
    s = spec_for(src.read_bytes(), url=src.as_uri())
    dest = tmp_path / "out" / "m.onnx"
    dm.download(s, dest)
    assert dest.read_bytes() == src.read_bytes()
    assert not list(dest.parent.glob("*.part"))


def test_download_rejects_tampered_or_truncated_file(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"tampered" * 10)
    s = spec_for(b"the real file" * 10, url=src.as_uri())
    dest = tmp_path / "out" / "m.onnx"
    with pytest.raises(RuntimeError, match="invalid"):
        dm.download(s, dest)
    assert not dest.exists() and not list(dest.parent.glob("*.part"))   # nothing half-installed


def test_app_starts_and_reports_clearly_when_models_missing(tmp_path, monkeypatch):
    from app import create_app
    from backend.services.face_recognition import EmbeddingStore
    monkeypatch.setattr(config, "DETECTOR_MODEL", tmp_path / "nope1.onnx")
    monkeypatch.setattr(config, "RECOGNIZER_MODEL", tmp_path / "nope2.onnx")
    client = create_app(store=EmbeddingStore(tmp_path / "e", legacy_dir=tmp_path / "l")).test_client()
    assert client.get("/health").status_code == 503            # process is up, but NOT ready: health must say so
    assert client.get("/").status_code == 200                  # container stays up and diagnosable
    assert client.get("/api/status").get_json()["engine_ready"] is False
    assert client.post("/api/recognize", json={"image": "x"}).status_code == 503
    assert client.post("/api/register", json={}).status_code == 503
