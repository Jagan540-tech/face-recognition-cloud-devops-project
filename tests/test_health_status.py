def test_health(env):
    client = env[0]
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json() == {"status": "healthy"}


def test_status(env):
    d = env[0].get("/api/status").get_json()
    assert d["engine_ready"] is True and d["registered_users"] == 0
    assert d["model"] == "sface_2021dec" and d["legacy_data_detected"] is False
    assert d["incompatible_store"] is None


def test_pages_render(env):
    for p in ("/", "/register", "/recognize"):
        assert env[0].get(p).status_code == 200


def test_unknown_route_is_json_404(env):
    r = env[0].get("/nope")
    assert r.status_code == 404 and r.get_json()["success"] is False


def test_health_is_unhealthy_when_models_cannot_load(tmp_path, monkeypatch):
    """No engine injected + missing model files -> app starts but /health must NOT say healthy."""
    import config
    from app import create_app
    from backend.services.face_recognition import EmbeddingStore
    monkeypatch.setattr(config, "DETECTOR_MODEL", tmp_path / "missing_detector.onnx")
    monkeypatch.setattr(config, "RECOGNIZER_MODEL", tmp_path / "missing_recognizer.onnx")
    store = EmbeddingStore(tmp_path / "embeddings", legacy_dir=tmp_path / "legacy")
    client = create_app(store=store, known_dir=tmp_path / "known").test_client()

    r = client.get("/health")
    assert r.status_code == 503
    assert r.get_json() == {"status": "unhealthy", "engine_ready": False}
    assert str(tmp_path) not in r.get_data(as_text=True)      # no file paths leak to the public

    assert client.get("/api/status").get_json()["engine_ready"] is False
    assert client.post("/api/recognize", json={"image": "x"}).status_code == 503
    assert client.get("/").status_code == 200                  # pages still render


def test_health_body_is_unchanged_when_ready(env):
    """Ansible and the Docker HEALTHCHECK only need HTTP 200; the body stays the same as before."""
    r = env[0].get("/health")
    assert r.status_code == 200 and r.get_json() == {"status": "healthy"}
