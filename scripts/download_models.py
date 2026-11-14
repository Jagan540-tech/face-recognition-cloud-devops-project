#!/usr/bin/env python3
"""Download and verify the pretrained ONNX models. Standard library only (no pip needed),
so the Docker build can run it in a throw-away stage.

    python scripts/download_models.py            # download anything missing, then verify
    python scripts/download_models.py --verify   # verify only, never touches the network

Source: the official OpenCV Zoo repository (https://github.com/opencv/opencv_zoo).
SHA-256 and size below are the values recorded in the repository's own Git-LFS pointer files
for these exact files, so a wrong/partial/tampered download is rejected, never loaded.

  YuNet  face_detection_yunet_2023mar.onnx      MIT license        (face detector)
  SFace  face_recognition_sface_2021dec.onnx    Apache-2.0 license (face embedding, 128-d)
"""
import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = [
    {"file": "face_detection_yunet_2023mar.onnx",
     "url": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
     "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
     "size": 232589},
    {"file": "face_recognition_sface_2021dec.onnx",
     "url": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
     "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
     "size": 38696353},
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path, spec):
    """Return None if the file is valid, else a human-readable problem."""
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size != spec["size"]:
        return f"wrong size ({size} bytes, expected {spec['size']})"
    digest = sha256_of(path)
    if digest != spec["sha256"]:
        return f"wrong SHA-256 ({digest[:16]}..., expected {spec['sha256'][:16]}...)"
    return None


def download(spec, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    try:
        print(f"  downloading {spec['file']} ...")
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "face-app-model-setup"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        problem = check(Path(tmp), spec)
        if problem:
            raise RuntimeError(f"downloaded file is invalid: {problem}")
        os.replace(tmp, dest)  # only a verified file ever gets the final name
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.environ.get("FR_MODELS_DIR",
                    str(Path(__file__).resolve().parent.parent / "models")))
    ap.add_argument("--verify", action="store_true", help="verify existing files only")
    args = ap.parse_args()
    target = Path(args.dir)
    failed = False
    for spec in MODELS:
        path = target / spec["file"]
        problem = check(path, spec)
        if problem and not args.verify:
            try:
                download(spec, path)
                problem = check(path, spec)
            except Exception as exc:
                print(f"FAILED  {spec['file']}: {exc}", file=sys.stderr)
                failed = True
                continue
        if problem:
            print(f"FAILED  {spec['file']}: {problem}", file=sys.stderr)
            failed = True
        else:
            print(f"OK      {spec['file']}  ({path.stat().st_size} bytes, sha256 verified)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
