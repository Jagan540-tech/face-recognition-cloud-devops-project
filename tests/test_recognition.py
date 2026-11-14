import config
from conftest import identity_vector, jpeg_data_url, make_face, register

A, B, C = (50, 50, 120, 140), (300, 50, 120, 140), (50, 300, 120, 140)


def rec(client, **kw):
    return client.post("/api/recognize", json={"image": jpeg_data_url(), **kw})


def test_invalid_payloads_are_400_not_500(env):
    client = env[0]
    for payload in ({"image": "garbage"}, {}, {"image": None}, {"image": 5}, [1]):
        assert client.post("/api/recognize", json=payload).status_code == 400
    assert client.post("/api/recognize", data="x", content_type="text/plain").status_code == 400


def test_no_faces_returns_empty_list(env):
    client, engine = env[0], env[1]
    engine.faces = []
    d = rec(client).get_json()
    assert d["success"] is True and d["faces"] == []


def test_unknown_when_store_is_empty(env):
    client, engine = env[0], env[1]
    engine.faces = [make_face(A, 1)]
    f = rec(client).get_json()["faces"][0]
    assert f["recognized"] is False and f["name"] == "Unknown" and f["role"] is None
    assert f["distance"] is None


def test_multi_face_each_face_independent(env):
    client, engine = env[0], env[1]
    register(client, engine, "Athul", "Student", 11)
    register(client, engine, "Rahul", "Developer", 22)
    engine.faces = [make_face(A, 11), make_face(B, 22), make_face(C, 33)]  # 33 = stranger
    faces = rec(client).get_json()["faces"]
    assert [f["name"] for f in faces] == ["Athul", "Rahul", "Unknown"]
    assert [f["recognized"] for f in faces] == [True, True, False]
    assert faces[0]["role"] == "Student" and faces[1]["role"] == "Developer"
    assert faces[0]["box"] == list(A) and faces[1]["box"] == list(B)
    assert faces[0]["similarity"] > 0.99 and abs(faces[0]["distance"]) < 0.01
    assert faces[2]["similarity"] < config.COSINE_THRESHOLD


def test_threshold_is_similarity_not_detection_confidence(env, monkeypatch):
    """A face whose cosine similarity is just below/above the threshold flips the decision."""
    client, engine, store, _ = env
    register(client, engine, "Athul", "Student", 11)
    import numpy as np
    base = identity_vector(11)
    other = identity_vector(12)
    for target_sim, expected in ((0.70, True), (0.50, False)):
        # build a unit vector with the requested cosine similarity to Athul's embedding
        ortho = other - base * float(base @ other)
        ortho /= np.linalg.norm(ortho)
        q = target_sim * base + np.sqrt(1 - target_sim ** 2) * ortho
        monkeypatch.setattr(engine, "embed", lambda f, face, q=q: q.astype(np.float32))
        engine.faces = [make_face(A, 11)]
        f = rec(client).get_json()["faces"][0]
        assert f["recognized"] is expected
        assert abs(f["similarity"] - target_sim) < 1e-3
        assert abs(f["distance"] - (1 - target_sim)) < 1e-3
        assert f["detection_score"] == 0.99          # reported separately, never mixed in


def test_boxes_are_scaled_back_to_original_frame(env, monkeypatch):
    client, engine = env[0], env[1]
    monkeypatch.setattr(config, "MAX_IMAGE_SIDE", 800)
    engine.faces = [make_face((100, 50, 60, 80), 1)]   # coordinates in the DOWNSCALED frame
    d = client.post("/api/recognize", json={"image": jpeg_data_url(size=(1200, 1600))}).get_json()
    assert engine.last_shape[:2] == (600, 800)          # server shrank 1600x1200 -> 800x600
    assert d["frame"] == {"width": 1600, "height": 1200}
    assert d["faces"][0]["box"] == [200, 100, 120, 160]


def test_engine_exception_returns_500_json(env, monkeypatch):
    client, engine = env[0], env[1]
    monkeypatch.setattr(engine, "detect", lambda f: (_ for _ in ()).throw(RuntimeError("boom")))
    r = rec(client)
    assert r.status_code == 500 and r.get_json()["success"] is False
