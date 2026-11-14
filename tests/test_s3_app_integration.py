"""The whole application with S3 as permanent storage (fake S3, fake face engine).

An "EC2 instance" is modelled by boot(): fresh, empty local directories + the SAME bucket.
Throwing the directories away and booting again is exactly what replacing an instance does.
"""
import io
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import MODEL_ID, FakeEngine, jpeg_data_url, make_face, register
from s3_fakes import FakeS3Client, client_error

import app as app_module
import config
from backend.services.s3_storage import S3Storage, S3StorageError

BOX = (100, 100, 120, 140)
ROOT = Path(__file__).resolve().parent.parent
PEOPLE = [("Athul", "Student", 1), ("Priya", "Engineer", 2), ("Rahul", "Manager", 3),
          ("Meera", "Designer", 4), ("Jagan", "DevOps", 5)]


@pytest.fixture
def s3():
    return FakeS3Client()


@pytest.fixture(autouse=True)
def three_samples(monkeypatch):
    monkeypatch.setattr(config, "SAMPLES_PER_PERSON", 3)


class Instance:
    def __init__(self, client, engine, root, storage):
        self.client, self.engine, self.root, self.storage = client, engine, root, storage


def boot(tmp_path, monkeypatch, s3, label):
    """Start the application the way gunicorn does on a (new) instance: create_app() with no store."""
    root = tmp_path / label
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", root / "emb")
    monkeypatch.setattr(config, "KNOWN_FACES_DIR", root / "known")
    monkeypatch.setattr(config, "LEGACY_DIR", root / "legacy")
    storage = S3Storage(bucket=s3.bucket, known_dir=root / "known", embeddings_dir=root / "emb", client=s3)
    engine = FakeEngine()
    flask_app = app_module.create_app(engine=engine, storage=storage)
    return Instance(flask_app.test_client(), engine, root, storage)


def recognize(inst, identity):
    inst.engine.faces = [make_face(BOX, identity)]
    r = inst.client.post("/api/recognize", json={"image": jpeg_data_url()})
    assert r.status_code == 200
    return r.get_json()["faces"][0]


def status(inst):
    return inst.client.get("/api/status").get_json()


# ------------------------------------------------------------------ startup
def test_empty_bucket_starts_with_no_registered_people(tmp_path, monkeypatch, s3):
    inst = boot(tmp_path, monkeypatch, s3, "i1")
    d = status(inst)
    assert d["registered_users"] == 0
    assert d["engine_ready"] is True
    assert "people" not in d
    assert inst.client.get("/health").status_code == 200
    assert recognize(inst, 1)["recognized"] is False


def test_s3_failure_at_startup_is_a_clear_error_not_a_silent_empty_start(tmp_path, monkeypatch, s3):
    s3.fail("list_objects_v2", client_error("AccessDenied", "ListObjectsV2"))
    with pytest.raises(S3StorageError, match="access denied"):
        boot(tmp_path, monkeypatch, s3, "i1")


def test_s3_is_downloaded_before_the_embedding_store_loads(tmp_path, monkeypatch, s3):
    first = boot(tmp_path, monkeypatch, s3, "i1")
    register(first.client, first.engine, "Athul", "Student", 1)

    events = []
    real_store = app_module.EmbeddingStore

    class SpyStore(real_store):
        def load(self):
            events.append("store.load")
            super().load()

    monkeypatch.setattr(app_module, "EmbeddingStore", SpyStore)
    real_restore = S3Storage.restore

    def spy_restore(self):
        events.append("s3.restore")
        return real_restore(self)
    monkeypatch.setattr(S3Storage, "restore", spy_restore)

    second = boot(tmp_path, monkeypatch, s3, "i2")
    assert events[:2] == ["s3.restore", "store.load"]
    assert status(second)["registered_users"] == 1
    assert recognize(second, 1)["name"] == "Athul"


def test_env_configured_bucket_is_used_when_app_builds_its_own_store(tmp_path, monkeypatch, s3):
    """The production path: FR_S3_BUCKET set, create_app() with no store and no storage."""
    import boto3
    seed = boot(tmp_path, monkeypatch, s3, "seed")
    register(seed.client, seed.engine, "Athul", "Student", 1)

    monkeypatch.setattr(config, "S3_BUCKET", s3.bucket)
    monkeypatch.setattr(boto3, "client", lambda *a, **k: s3)
    root = tmp_path / "prod"
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", root / "emb")
    monkeypatch.setattr(config, "KNOWN_FACES_DIR", root / "known")
    monkeypatch.setattr(config, "LEGACY_DIR", root / "legacy")
    client = app_module.create_app(engine=FakeEngine()).test_client()
    assert client.get("/api/status").get_json()["registered_users"] == 1


# ------------------------------------------------------------------ S3 never touched by tests / imports
def test_importing_the_app_never_imports_boto3_or_connects(tmp_path):
    code = ("import sys; import app; "
            "assert 'boto3' not in sys.modules, 'importing the app must not even import boto3'; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": "", "FR_S3_BUCKET": "some-bucket", "HOME": str(tmp_path),
                              "AWS_EC2_METADATA_DISABLED": "true", "PYTHONPATH": str(ROOT)})
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_injected_store_means_no_s3_even_when_a_bucket_is_configured(tmp_path, monkeypatch):
    import boto3
    from backend.services.face_recognition import EmbeddingStore
    monkeypatch.setattr(config, "S3_BUCKET", "configured-bucket")

    def boom(*a, **k):
        raise AssertionError("S3 must not be used when a store is injected")
    monkeypatch.setattr(boto3, "client", boom)
    store = EmbeddingStore(tmp_path / "emb", legacy_dir=tmp_path / "legacy")
    client = app_module.create_app(engine=FakeEngine(), store=store, known_dir=tmp_path / "known").test_client()
    assert client.get("/health").status_code == 200


# ------------------------------------------------------------------ registration -> S3
def test_registration_uploads_photos_metadata_and_embeddings(tmp_path, monkeypatch, s3):
    inst = boot(tmp_path, monkeypatch, s3, "i1")
    register(inst.client, inst.engine, "Athul", "Student", 1)

    keys = sorted(s3.objects)
    assert [k for k in keys if k.endswith(".jpg")] and len([k for k in keys if k.endswith(".jpg")]) == 3
    assert "known_faces/Athul/metadata.json" in keys and "embeddings/embeddings.npz" in keys
    local = (inst.root / "emb" / "embeddings.npz").read_bytes()
    assert s3.objects["embeddings/embeddings.npz"] == local       # existing format, byte for byte
    with np.load(io.BytesIO(local), allow_pickle=False) as z:
        assert str(z["model_id"]) == MODEL_ID and z["embeddings"].shape == (3, 128)


def test_success_is_only_reported_after_the_embeddings_reached_s3(tmp_path, monkeypatch, s3):
    inst = boot(tmp_path, monkeypatch, s3, "i1")
    s3.fail("upload_file", client_error("500", "PutObject"), key_contains="embeddings/")
    inst.engine.faces = [make_face(BOX, 7)]
    body = {"session_id": "s1", "name": "Athul", "role": "Student"}
    for _ in range(3):
        inst.client.post("/api/register", json={**body, "image": jpeg_data_url()})

    r = inst.client.post("/api/register", json={**body, "finish": True})
    assert r.status_code == 503 and r.get_json()["success"] is False
    assert "permanent storage" in r.get_json()["error"]
    assert status(inst)["registered_users"] == 0                     # rolled back: memory == S3
    assert recognize(inst, 7)["recognized"] is False
    assert "embeddings/embeddings.npz" not in s3.objects

    s3.heal()                                                        # S3 is back: finish can be retried
    r = inst.client.post("/api/register", json={**body, "finish": True})
    assert r.status_code == 200 and r.get_json()["success"] is True
    assert status(inst)["registered_users"] == 1
    assert "embeddings/embeddings.npz" in s3.objects
    assert recognize(inst, 7)["name"] == "Athul"


def test_failed_photo_upload_registers_nobody(tmp_path, monkeypatch, s3):
    inst = boot(tmp_path, monkeypatch, s3, "i1")
    s3.fail("upload_file", client_error("500", "PutObject"), key_contains=".jpg")
    inst.engine.faces = [make_face(BOX, 7)]
    body = {"session_id": "s1", "name": "Athul", "role": "Student"}
    for _ in range(3):
        inst.client.post("/api/register", json={**body, "image": jpeg_data_url()})
    r = inst.client.post("/api/register", json={**body, "finish": True})
    assert r.status_code == 503
    assert status(inst)["registered_users"] == 0
    assert "known_faces/Athul/metadata.json" not in s3.objects and "embeddings/embeddings.npz" not in s3.objects


def test_worker_restart_discards_local_leftovers_of_a_failed_registration(tmp_path, monkeypatch, s3):
    """First-ever registration fails at the embeddings upload; the worker then restarts in the SAME
    container. The stale local embeddings.npz must not resurrect a person S3 never received."""
    inst = boot(tmp_path, monkeypatch, s3, "i1")
    s3.fail("upload_file", client_error("500", "PutObject"), key_contains="embeddings/")
    inst.engine.faces = [make_face(BOX, 7)]
    body = {"session_id": "s1", "name": "Athul", "role": "Student"}
    for _ in range(3):
        inst.client.post("/api/register", json={**body, "image": jpeg_data_url()})
    assert inst.client.post("/api/register", json={**body, "finish": True}).status_code == 503
    assert (inst.root / "emb" / "embeddings.npz").exists()           # local scratch has it...

    s3.heal()
    again = boot(tmp_path, monkeypatch, s3, "i1")                    # ...same directories, restarted
    assert status(again)["registered_users"] == 0


# ------------------------------------------------------------------ THE acceptance scenario
def test_ec2_replacement_five_people_survive_and_are_recognised(tmp_path, monkeypatch, s3):
    """1. instance runs  2. register five people  3. data is in S3  4. instance terminated
    5. new instance + same bucket  6. all five restored  7. webcam recognises each of them."""
    old = boot(tmp_path, monkeypatch, s3, "old-instance")
    for name, role, identity in PEOPLE:
        register(old.client, old.engine, name, role, identity)
    assert status(old)["registered_users"] == 5

    shutil.rmtree(old.root)                                          # "terminate" - nothing local survives
    assert not old.root.exists()

    new = boot(tmp_path, monkeypatch, s3, "new-instance")            # fresh disk, same bucket
    d = status(new)
    assert d["registered_users"] == 5
    assert d["engine_ready"] is True
    assert "people" not in d
    for name, role, identity in PEOPLE:
        face = recognize(new, identity)
        assert (face["recognized"], face["name"], face["role"]) == (True, name, role)
        folder = new.root / "known" / name                           # photos + metadata restored too
        assert len(list(folder.glob("*.jpg"))) == 3 and (folder / "metadata.json").exists()
    assert recognize(new, 99)["recognized"] is False                 # a stranger is still unknown


def test_any_number_of_people_each_registration_keeps_all_earlier_ones_in_s3(tmp_path, monkeypatch, s3):
    inst = boot(tmp_path, monkeypatch, s3, "i1")
    for i, (name, role, identity) in enumerate(PEOPLE, start=1):
        register(inst.client, inst.engine, name, role, identity)
        with np.load(io.BytesIO(s3.objects["embeddings/embeddings.npz"]), allow_pickle=False) as z:
            assert len(set(z["names"].tolist())) == i and z["embeddings"].shape[0] == 3 * i


def test_reregistering_a_name_replaces_it_in_s3_too(tmp_path, monkeypatch, s3):
    inst = boot(tmp_path, monkeypatch, s3, "i1")
    register(inst.client, inst.engine, "Athul", "Student", 1)
    register(inst.client, inst.engine, "Athul", "Lead", 8)
    after = boot(tmp_path, monkeypatch, s3, "i2")
    assert status(after)["registered_users"] == 1
    assert recognize(after, 8)["role"] == "Lead" and recognize(after, 1)["recognized"] is False


# ------------------------------------------------------------------ recognition stays in RAM
def test_recognition_makes_no_s3_requests(tmp_path, monkeypatch, s3):
    old = boot(tmp_path, monkeypatch, s3, "i1")
    for name, role, identity in PEOPLE[:2]:
        register(old.client, old.engine, name, role, identity)
    new = boot(tmp_path, monkeypatch, s3, "i2")
    calls_after_startup = len(s3.calls)
    assert calls_after_startup > 0                                    # startup did talk to S3

    for _ in range(25):                                               # 25 "webcam frames"
        assert recognize(new, 1)["recognized"] is True
        assert recognize(new, 99)["recognized"] is False
        new.client.get("/api/status")
    assert len(s3.calls) == calls_after_startup                       # ...and recognition never did
