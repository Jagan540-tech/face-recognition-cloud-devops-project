"""Amazon S3 permanent storage for registration data.

S3 layout (bucket is private, created outside this project):

    known_faces/<Name>/<timestamp>_<n>.jpg   registration photos
    known_faces/<Name>/metadata.json         written LAST, so a folder that has it is complete
    embeddings/embeddings.npz                the existing embedding store, byte for byte

What this class does - and nothing more:
  * check_access()        one cheap call that proves the bucket is reachable and listable
  * restore()             make the local scratch dirs an exact copy of S3 (run BEFORE EmbeddingStore loads)
  * upload_person()       photos, then metadata.json
  * upload_embeddings()   the updated embeddings.npz

Credentials: boto3's default chain only. On EC2 that is the instance role (FaceRecognitionEC2Role);
no access key is ever read, written or logged by this project.

boto3 is imported lazily and a client is created on first use, so importing this module (or the
Flask app) never touches the network. Nothing here is called per webcam frame: recognition runs
entirely from the embeddings already in RAM.

Logs and error messages never contain names or object keys (they contain personal data).
"""
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import config

log = logging.getLogger(__name__)

WORKERS = 8          # parallel transfers (many small photos); boto3 clients are thread-safe
METADATA_FILE = "metadata.json"


class S3StorageError(RuntimeError):
    """Any S3 failure, already phrased so it can be shown in a log or to a user."""


def _explain(exc):
    """Turn a boto3/botocore exception into one actionable sentence (no keys, no names)."""
    name = type(exc).__name__
    code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
    if name in ("NoCredentialsError", "PartialCredentialsError", "CredentialRetrievalError"):
        return ("no AWS credentials found. On EC2 attach the IAM role FaceRecognitionEC2Role to the "
                "instance; inside Docker also make sure the instance metadata hop limit is at least 2")
    if code in ("AccessDenied", "AllAccessDisabled", "403"):
        return ("access denied. The instance role needs s3:ListBucket on the bucket and "
                "s3:GetObject / s3:PutObject on its objects")
    if code in ("NoSuchBucket", "404"):
        return "the bucket does not exist (check FR_S3_BUCKET)"
    if code in ("InvalidAccessKeyId", "SignatureDoesNotMatch", "ExpiredToken"):
        return f"AWS rejected the credentials ({code})"
    if name in ("EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError", "ConnectionClosedError"):
        return "could not reach Amazon S3 (network / timeout)"
    return f"{name}: {exc}"


class S3Storage:
    def __init__(self, bucket=None, region=None, known_dir=None, embeddings_dir=None, client=None):
        self.bucket = bucket or config.S3_BUCKET
        self.region = region or config.S3_REGION
        self.known_dir = Path(known_dir) if known_dir is not None else config.KNOWN_FACES_DIR
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir is not None else config.EMBEDDINGS_DIR
        if not self.bucket:
            raise S3StorageError("FR_S3_BUCKET is not set")
        self._client = client  # tests inject a fake; production creates one lazily

    # ------------------------------------------------------------------ plumbing
    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:  # pragma: no cover - boto3 is in requirements.txt
                raise S3StorageError("boto3 is not installed") from exc
            self._client = boto3.client(
                "s3", region_name=self.region,
                config=Config(retries={"max_attempts": 4, "mode": "standard"},
                              connect_timeout=5, read_timeout=30))
        return self._client

    def _run(self, what, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except S3StorageError:
            raise
        except Exception as exc:
            raise S3StorageError(f"{what} failed for bucket '{self.bucket}': {_explain(exc)}") from exc

    def _parallel(self, fn, items):
        """Run fn(*item) for every item; the first failure is raised."""
        items = list(items)
        if not items:
            return
        self.client  # create the client once, before the worker threads start
        with ThreadPoolExecutor(max_workers=min(WORKERS, len(items))) as pool:
            for future in [pool.submit(fn, *item) for item in items]:
                future.result()

    # ------------------------------------------------------------------ access check
    def check_access(self):
        """Proves the bucket exists, credentials work and listing is allowed. Raises S3StorageError."""
        self._run("S3 access check", self.client.list_objects_v2, Bucket=self.bucket, MaxKeys=1)

    # ------------------------------------------------------------------ restore (startup)
    def _list(self):
        """[(key, local_path)] for everything under both prefixes. Pure listing: touches nothing local."""
        found = []
        for prefix, base in ((config.S3_KNOWN_FACES_PREFIX, self.known_dir),
                             (config.S3_EMBEDDINGS_PREFIX, self.embeddings_dir)):
            def collect(prefix=prefix, base=base):
                pages = self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix)
                for page in pages:
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        if key.endswith("/"):      # console-created "folder" placeholder
                            continue
                        found.append((key, self._local_path(prefix, base, key)))
            self._run("Listing registration data", collect)
        return found

    @staticmethod
    def _local_path(prefix, base, key):
        parts = PurePosixPath(key[len(prefix):]).parts
        if not parts or any(p in ("", ".", "..") for p in parts) or "\\" in key:
            raise S3StorageError("refusing an unsafe object key found in the bucket")
        dest = Path(base).joinpath(*parts)
        if Path(base).resolve() not in dest.resolve().parents:
            raise S3StorageError("refusing an object key that points outside the data directory")
        return dest

    def _clear_local(self):
        """Local data is a scratch copy: remove what S3 does not know about (e.g. a half-finished
        registration whose upload failed) so the local copy is exactly S3."""
        if self.known_dir.is_dir():
            for child in self.known_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
        if self.embeddings_dir.is_dir():
            for stale in list(self.embeddings_dir.glob("*.npz")) + list(self.embeddings_dir.glob("*.tmp")):
                stale.unlink()

    def _download(self, key, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._run("Downloading a registration file", self.client.download_file, self.bucket, key, str(dest))

    def restore(self):
        """Download everything from S3 into the local scratch dirs. Call BEFORE EmbeddingStore loads.

        An empty bucket is fine (the app starts with no registered people). Any error raises
        S3StorageError, so the application refuses to start rather than silently running with
        missing registration data. The listing happens first: if S3 is unreachable nothing local
        is touched.
        """
        self.check_access()
        objects = self._list()
        self._clear_local()
        self._parallel(self._download, objects)
        people = {k.split("/")[1] for k, _ in objects
                  if k.startswith(config.S3_KNOWN_FACES_PREFIX) and k.count("/") >= 2}
        log.info("Restored %d files (%d registered people) from S3 bucket '%s'",
                 len(objects), len(people), self.bucket)
        return {"files": len(objects), "people": len(people)}

    # ------------------------------------------------------------------ upload (registration)
    def _upload(self, path, key):
        self._run("Uploading registration data", self.client.upload_file, str(path), self.bucket, key)

    def upload_person(self, folder):
        """Photos first (in parallel), metadata.json last: a folder in S3 that has metadata.json is complete."""
        folder = Path(folder)
        prefix = f"{config.S3_KNOWN_FACES_PREFIX}{folder.name}/"
        files = sorted(p for p in folder.iterdir() if p.is_file())
        self._parallel(self._upload, [(p, prefix + p.name) for p in files if p.name != METADATA_FILE])
        for p in files:
            if p.name == METADATA_FILE:
                self._upload(p, prefix + p.name)

    def upload_embeddings(self, path):
        """Upload the (already atomically written) embeddings file."""
        path = Path(path)
        self._upload(path, f"{config.S3_EMBEDDINGS_PREFIX}{path.name}")
