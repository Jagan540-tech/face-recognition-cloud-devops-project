import numpy as np
from conftest import identity_vector

from backend.services.face_recognition import EmbeddingStore


def make(tmp, **kw):
    return EmbeddingStore(tmp / "emb", legacy_dir=tmp / "legacy", **kw)


def test_roundtrip_persists_across_restart(tmp_path):
    s = make(tmp_path)
    s.add([identity_vector(1), identity_vector(1)], "Athul", "Student")
    s2 = make(tmp_path)
    assert s2.people() == ["Athul"] and len(s2.snapshot()[0]) == 2 and s2.incompatible is None


def test_same_dimension_different_model_is_refused_and_preserved(tmp_path):
    """Both models output 128 numbers - they must still never be compared."""
    old = make(tmp_path, model_id="some_other_128d_model")
    old.add([identity_vector(1)], "Athul", "Student")
    path = old.path
    before = path.read_bytes()

    s = make(tmp_path)                                   # now running the SFace id
    assert s.incompatible and "some_other_128d_model" in s.incompatible
    assert s.people() == [] and len(s.snapshot()[0]) == 0   # nothing loaded -> nothing compared
    assert path.read_bytes() == before                   # file untouched by loading

    s.add([identity_vector(2)], "Rahul", "Dev")          # registering must not destroy old data
    backups = list(path.parent.glob("embeddings.incompatible-*.npz"))
    assert len(backups) == 1 and backups[0].read_bytes() == before
    assert s.people() == ["Rahul"] and s.incompatible is None


def test_dimension_mismatch_refused(tmp_path):
    other = make(tmp_path, model_id="sface_2021dec", dim=64)
    other.add([np.ones(64, np.float32)], "X", "Y")
    s = make(tmp_path)
    assert s.incompatible and s.people() == []


def test_corrupt_store_is_left_untouched(tmp_path):
    (tmp_path / "emb").mkdir()
    p = tmp_path / "emb" / "embeddings.npz"
    p.write_bytes(b"not a zip")
    s = make(tmp_path)
    assert s.incompatible and "unreadable" in s.incompatible
    assert p.read_bytes() == b"not a zip"


def test_legacy_dlib_files_detected_never_loaded_never_modified(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    legacy_vecs = np.random.default_rng(0).normal(size=(5, 128))   # 128-d, like SFace!
    np.save(legacy / "encodings.npy", legacy_vecs)
    np.save(legacy / "names.npy", np.array(["Jagan"] * 5))
    np.save(legacy / "details.npy", np.array(["Dev"] * 5))
    snapshot = {p.name: p.read_bytes() for p in legacy.iterdir()}

    s = make(tmp_path)
    assert s.legacy_detected() is True
    assert s.people() == [] and len(s.snapshot()[0]) == 0           # not used as embeddings
    s.add([identity_vector(3)], "Athul", "Student")
    assert {p.name: p.read_bytes() for p in legacy.iterdir()} == snapshot   # untouched


def test_status_endpoint_reports_legacy_and_incompatible(env):
    client, engine, store, tmp = env
    legacy = tmp / "legacy"
    legacy.mkdir()
    for n in ("encodings", "names", "details"):
        np.save(legacy / f"{n}.npy", np.zeros(1))
    store.load()
    d = client.get("/api/status").get_json()
    assert d["legacy_data_detected"] is True
