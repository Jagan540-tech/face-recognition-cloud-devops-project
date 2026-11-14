"""S3Storage on its own, against an in-memory fake client (no AWS credentials, no network)."""
import json

import pytest
from botocore.exceptions import NoCredentialsError
from s3_fakes import FakeS3Client, client_error

import config
from backend.services.s3_storage import S3Storage, S3StorageError


def make(tmp_path, client=None):
    client = client or FakeS3Client()
    storage = S3Storage(bucket="test-bucket", region="ap-south-1", known_dir=tmp_path / "known",
                        embeddings_dir=tmp_path / "emb", client=client)
    return storage, client


def put_person(client, name, photos=3):
    for i in range(photos):
        client.objects[f"known_faces/{name}/2026_{i}.jpg"] = b"jpeg-bytes-" + str(i).encode()
    client.objects[f"known_faces/{name}/metadata.json"] = json.dumps({"name": name}).encode()


# ------------------------------------------------------------------ restore
def test_empty_bucket_restores_nothing_and_does_not_fail(tmp_path):
    storage, _ = make(tmp_path)
    assert storage.restore() == {"files": 0, "people": 0}


def test_restore_downloads_photos_metadata_and_embeddings(tmp_path):
    storage, client = make(tmp_path)
    put_person(client, "Athul")
    put_person(client, "Priya")          # 4 keys per person -> several listing pages
    client.objects["embeddings/embeddings.npz"] = b"npz-bytes"

    summary = storage.restore()

    assert summary == {"files": 9, "people": 2}
    for name in ("Athul", "Priya"):
        folder = tmp_path / "known" / name
        assert len(list(folder.glob("*.jpg"))) == 3
        assert json.loads((folder / "metadata.json").read_text())["name"] == name
    assert (tmp_path / "emb" / "embeddings.npz").read_bytes() == b"npz-bytes"


def test_restore_makes_local_an_exact_copy_of_s3(tmp_path):
    """Anything local that S3 does not have (e.g. a half-finished registration) must go."""
    storage, client = make(tmp_path)
    put_person(client, "Athul")
    (tmp_path / "known" / "Ghost").mkdir(parents=True)
    (tmp_path / "known" / "Ghost" / "a.jpg").write_bytes(b"x")
    (tmp_path / "known" / ".gitkeep").write_bytes(b"")
    (tmp_path / "emb").mkdir()
    (tmp_path / "emb" / "embeddings.npz").write_bytes(b"stale local store")

    storage.restore()

    assert not (tmp_path / "known" / "Ghost").exists()
    assert (tmp_path / "known" / ".gitkeep").exists()          # non-person files are left alone
    assert not (tmp_path / "emb" / "embeddings.npz").exists()  # S3 has none -> none locally


def test_restore_ignores_console_folder_placeholders(tmp_path):
    storage, client = make(tmp_path)
    client.objects["known_faces/"] = b""
    client.objects["embeddings/"] = b""
    assert storage.restore() == {"files": 0, "people": 0}


def test_failed_listing_leaves_local_data_untouched(tmp_path):
    storage, client = make(tmp_path)
    (tmp_path / "known" / "Athul").mkdir(parents=True)
    (tmp_path / "known" / "Athul" / "a.jpg").write_bytes(b"x")
    client.fail("list_objects_v2", client_error("AccessDenied"))
    with pytest.raises(S3StorageError):
        storage.restore()
    assert (tmp_path / "known" / "Athul" / "a.jpg").exists()


def test_download_failure_is_reported(tmp_path):
    storage, client = make(tmp_path)
    put_person(client, "Athul")
    client.fail("download_file", client_error("500"), key_contains=".jpg")
    with pytest.raises(S3StorageError, match="Downloading a registration file failed"):
        storage.restore()


@pytest.mark.parametrize("bad_key", ["known_faces/../evil.txt", "known_faces/a/../../evil.txt",
                                     "known_faces//x.jpg", "known_faces/a\\..\\b.jpg"])
def test_unsafe_keys_are_refused(tmp_path, bad_key):
    storage, client = make(tmp_path)
    client.objects[bad_key] = b"x"
    with pytest.raises(S3StorageError, match="unsafe|outside"):
        storage.restore()
    assert not (tmp_path / "evil.txt").exists()


# ------------------------------------------------------------------ uploads
def test_upload_person_uploads_photos_then_metadata_last(tmp_path):
    storage, client = make(tmp_path)
    folder = tmp_path / "known" / "Athul"
    folder.mkdir(parents=True)
    for i in range(5):
        (folder / f"p{i}.jpg").write_bytes(b"jpg" + bytes([i]))
    (folder / "metadata.json").write_text('{"name": "Athul"}')

    storage.upload_person(folder)

    assert sorted(client.objects) == sorted([f"known_faces/Athul/p{i}.jpg" for i in range(5)]
                                            + ["known_faces/Athul/metadata.json"])
    uploads = [key for op, key in client.calls if op == "upload_file"]
    assert uploads[-1] == "known_faces/Athul/metadata.json"   # a folder with metadata.json is complete


def test_upload_embeddings_uses_embeddings_prefix(tmp_path):
    storage, client = make(tmp_path)
    f = tmp_path / "embeddings.npz"
    f.write_bytes(b"vectors")
    storage.upload_embeddings(f)
    assert client.objects == {"embeddings/embeddings.npz": b"vectors"}


def test_failed_photo_upload_never_uploads_metadata(tmp_path):
    storage, client = make(tmp_path)
    folder = tmp_path / "known" / "Athul"
    folder.mkdir(parents=True)
    (folder / "p0.jpg").write_bytes(b"j")
    (folder / "metadata.json").write_text("{}")
    client.fail("upload_file", client_error("500"), key_contains=".jpg")
    with pytest.raises(S3StorageError):
        storage.upload_person(folder)
    assert "known_faces/Athul/metadata.json" not in client.objects


# ------------------------------------------------------------------ clear error messages
def test_access_denied_message_names_the_needed_permissions(tmp_path):
    storage, client = make(tmp_path)
    client.fail("list_objects_v2", client_error("AccessDenied", "ListObjectsV2"))
    with pytest.raises(S3StorageError) as e:
        storage.check_access()
    assert "s3:ListBucket" in str(e.value) and "test-bucket" in str(e.value)


def test_no_credentials_message_points_at_the_instance_role(tmp_path):
    storage, client = make(tmp_path)
    client.fail("list_objects_v2", NoCredentialsError())
    with pytest.raises(S3StorageError) as e:
        storage.check_access()
    assert "FaceRecognitionEC2Role" in str(e.value) and "hop limit" in str(e.value)


def test_missing_bucket_message(tmp_path):
    storage, client = make(tmp_path)
    client.fail("list_objects_v2", client_error("NoSuchBucket"))
    with pytest.raises(S3StorageError, match="does not exist"):
        storage.check_access()


def test_errors_and_logs_do_not_contain_names(tmp_path, caplog):
    storage, client = make(tmp_path)
    folder = tmp_path / "known" / "Athul"
    folder.mkdir(parents=True)
    (folder / "p0.jpg").write_bytes(b"j")
    client.fail("upload_file", client_error("500"))
    with caplog.at_level("DEBUG"):
        with pytest.raises(S3StorageError) as e:
            storage.upload_person(folder)
    assert "Athul" not in str(e.value) and "Athul" not in caplog.text


def test_bucket_is_required(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "")
    with pytest.raises(S3StorageError, match="FR_S3_BUCKET"):
        S3Storage(known_dir=tmp_path, embeddings_dir=tmp_path)


def test_constructing_storage_creates_no_boto3_client(tmp_path, monkeypatch):
    import boto3

    def boom(*a, **k):
        raise AssertionError("boto3.client() must not be called until S3 is actually used")
    monkeypatch.setattr(boto3, "client", boom)
    S3Storage(bucket="b", known_dir=tmp_path, embeddings_dir=tmp_path)   # must not raise
