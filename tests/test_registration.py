import numpy as np
from conftest import MODEL_ID, jpeg_data_url, make_face

BOX = (100, 100, 120, 140)


def post(client, **kw):
    body = {"session_id": "s1", "name": "Athul", "role": "Student", **kw}
    return client.post("/api/register", json=body)


def test_empty_name_or_role_rejected(env):
    client = env[0]
    assert post(client, name="  ", image=jpeg_data_url()).status_code == 400
    assert post(client, role="", image=jpeg_data_url()).status_code == 400


def test_malformed_requests_do_not_crash(env):
    client = env[0]
    assert client.post("/api/register", json=[1, 2]).status_code == 400
    assert client.post("/api/register", data="not json", content_type="text/plain").status_code == 400
    assert post(client, name=123, image=jpeg_data_url()).status_code == 400
    assert post(client, session_id="../../etc").status_code == 400
    assert post(client, session_id=None).status_code == 400


def test_invalid_image_not_counted(env):
    client, engine = env[0], env[1]
    engine.faces = [make_face(BOX, 1)]
    d = post(client, image="data:image/jpeg;base64,AAAA").get_json()
    assert d["accepted"] is False and d["code"] == "bad_image"
    assert post(client, image=12345).get_json()["code"] == "bad_image"


def test_no_face_not_counted(env):
    client, engine = env[0], env[1]
    engine.faces = []
    d = post(client, image=jpeg_data_url()).get_json()
    assert (d["accepted"], d["code"], d["count"]) == (False, "no_face", 0)


def test_multiple_faces_not_counted(env):
    client, engine = env[0], env[1]
    engine.faces = [make_face(BOX, 1), make_face((300, 100, 120, 140), 2)]
    assert post(client, image=jpeg_data_url()).get_json()["code"] == "multiple"


def test_small_face_not_counted(env):
    client, engine = env[0], env[1]
    engine.faces = [make_face((10, 10, 20, 20), 1)]
    assert post(client, image=jpeg_data_url()).get_json()["code"] == "too_small"


def test_full_registration_saves_everything(env):
    client, engine, store, tmp = env
    engine.faces = [make_face(BOX, 7)]
    for i in range(3):
        d = post(client, image=jpeg_data_url()).get_json()
        assert d["accepted"] and d["count"] == i + 1
    d = post(client, image=jpeg_data_url()).get_json()          # 4th sample is refused
    assert d["accepted"] is False and d["code"] == "complete"
    j = post(client, finish=True).get_json()
    assert (j["name"], j["role"], j["samples"]) == ("Athul", "Student", 3)
    assert store.people() == ["Athul"]
    # photos + metadata + embeddings on disk, embeddings carry the model id
    folder = tmp / "known" / "Athul"
    assert len(list(folder.glob("*.jpg"))) == 3 and (folder / "metadata.json").exists()
    with np.load(tmp / "embeddings" / "embeddings.npz", allow_pickle=False) as z:
        assert str(z["model_id"]) == MODEL_ID and z["embeddings"].shape == (3, 128)
        assert np.allclose(np.linalg.norm(z["embeddings"], axis=1), 1.0, atol=1e-5)


def test_finish_too_early_fails(env):
    client, engine = env[0], env[1]
    engine.faces = [make_face(BOX, 1)]
    post(client, image=jpeg_data_url())
    assert post(client, finish=True).status_code == 400


def test_reregistering_same_name_replaces_embeddings(env):
    client, engine, store, _ = env
    from conftest import register
    register(client, engine, "Athul", "Student", 7)
    register(client, engine, "Athul", "Lead", 8)
    emb, names, roles = store.snapshot()
    assert len(emb) == 3 and set(roles) == {"Lead"}
