"""In-memory stand-in for boto3's S3 client, used by the S3 tests.

It implements exactly the calls backend/services/s3_storage.py makes (list_objects_v2, the
list_objects_v2 paginator, upload_file, download_file), records every call so tests can assert on
S3 traffic, and can be told to fail so error paths are exercised. It raises the real botocore
exception classes, so the production error handling is what gets tested. No AWS credentials,
network or extra test dependency is needed.
"""
from pathlib import Path

from botocore.exceptions import ClientError

PAGE_SIZE = 2  # tiny on purpose: forces pagination in every listing


def client_error(code, operation="Operation"):
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


class FakeS3Client:
    def __init__(self, bucket="test-bucket"):
        self.bucket = bucket
        self.objects = {}   # key -> bytes
        self.calls = []     # (operation, key-or-prefix)
        self._failures = []  # (operation, key_substring_or_None, exception)

    # ------------------------------------------------------------ test controls
    def fail(self, operation, exc, key_contains=None):
        """Make `operation` raise `exc` (optionally only for keys containing `key_contains`)."""
        self._failures.append((operation, key_contains, exc))

    def heal(self):
        self._failures.clear()

    def calls_to(self, operation):
        return [c for c in self.calls if c[0] == operation]

    def _check(self, operation, key):
        self.calls.append((operation, key))
        for op, contains, exc in self._failures:
            if op == operation and (contains is None or contains in key):
                raise exc

    # ------------------------------------------------------------ the S3 API subset
    def list_objects_v2(self, Bucket, Prefix="", MaxKeys=1000, **_):
        self._check("list_objects_v2", Prefix)
        keys = sorted(k for k in self.objects if k.startswith(Prefix))[:MaxKeys]
        return {"Contents": [{"Key": k} for k in keys], "KeyCount": len(keys)}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self)

    def upload_file(self, Filename, Bucket, Key, **_):
        self._check("upload_file", Key)
        self.objects[Key] = Path(Filename).read_bytes()

    def download_file(self, Bucket, Key, Filename, **_):
        self._check("download_file", Key)
        if Key not in self.objects:
            raise client_error("404", "HeadObject")
        Path(Filename).write_bytes(self.objects[Key])


class _Paginator:
    def __init__(self, client):
        self.client = client

    def paginate(self, Bucket, Prefix=""):
        self.client._check("list_objects_v2", Prefix)
        keys = sorted(k for k in self.client.objects if k.startswith(Prefix))
        if not keys:
            yield {"KeyCount": 0}   # real S3 omits "Contents" for an empty listing
            return
        for i in range(0, len(keys), PAGE_SIZE):
            yield {"Contents": [{"Key": k} for k in keys[i:i + PAGE_SIZE]]}
