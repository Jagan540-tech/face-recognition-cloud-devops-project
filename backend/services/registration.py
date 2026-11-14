"""Registration: validate a webcam sample, save the photo, and on finish add the embeddings
to the live store (replaces running capture + train scripts by hand).

With S3 storage configured (see s3_storage.py) finish() also uploads the photos, metadata and
the updated embeddings to S3 and only reports success once that upload has worked."""
import json
import logging
import re
import threading
import time
from datetime import datetime

import cv2

import config

log = logging.getLogger(__name__)

MESSAGES = {
    "no_face": "No face detected. Please position your face inside the guide.",
    "multiple": "Multiple faces detected. Please make sure only one person is visible.",
    "too_small": "Face is too small. Please move closer to the camera.",
    "bad_image": "Could not read the image. Please try again.",
}
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")


def valid_session_id(value):
    return isinstance(value, str) and bool(SESSION_RE.match(value))


class Registrar:
    def __init__(self, engine, store, known_dir=None, storage=None):
        self.engine, self.store = engine, store
        self.known_dir = known_dir or config.KNOWN_FACES_DIR
        self.storage = storage  # S3Storage or None (None = local-only, the original behaviour)
        self.sessions = {}  # session_id -> state; in-memory, so run ONE gunicorn worker
        self.lock = threading.Lock()
        self._persist_lock = threading.Lock()  # one "add + upload embeddings" at a time

    def validate_person(self, name, role):
        if not isinstance(name, str) or not isinstance(role, str):
            return "Name and role must be text."
        name, role = name.strip(), role.strip()
        if not name:
            return "Name cannot be empty."
        if not role:
            return "Role cannot be empty."
        if not safe_name(name):
            return "Name must contain letters or numbers."
        if len(name) > 60 or len(role) > 60:
            return "Name and role must be 60 characters or fewer."
        return None

    def _expire_sessions(self):
        cutoff = time.time() - config.SESSION_TTL_SECONDS
        for sid in [k for k, v in self.sessions.items() if v["touched"] < cutoff]:
            self.sessions.pop(sid, None)

    def add_sample(self, session_id, name, role, frame):
        """Returns (accepted, code, message, state)."""
        err = self.validate_person(name, role)
        if err:
            return False, "invalid_person", err, None
        if frame is None:
            return False, "bad_image", MESSAGES["bad_image"], self._state(session_id)

        faces = self.engine.detect(frame)
        if not faces:
            return False, "no_face", MESSAGES["no_face"], self._state(session_id)
        if len(faces) > 1:
            return False, "multiple", MESSAGES["multiple"], self._state(session_id)
        face = faces[0]
        if face.box[2] < config.MIN_FACE_SIZE or face.box[3] < config.MIN_FACE_SIZE:
            return False, "too_small", MESSAGES["too_small"], self._state(session_id)

        try:
            vec = self.engine.embed(frame, face)
        except (ValueError, cv2.error):
            return False, "bad_image", MESSAGES["bad_image"], self._state(session_id)

        with self.lock:
            self._expire_sessions()
            s = self.sessions.get(session_id)
            if s is None or s["name"] != name.strip():
                folder = self.known_dir / safe_name(name)
                folder.mkdir(parents=True, exist_ok=True)
                s = {"name": name.strip(), "role": role.strip(), "dir": folder,
                     "embeddings": [], "count": 0, "touched": time.time()}
                self.sessions[session_id] = s
            s["touched"] = time.time()
            if s["count"] >= config.SAMPLES_PER_PERSON:
                return False, "complete", "Enough samples captured.", self._state(session_id)
            s["count"] += 1
            s["embeddings"].append(vec)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(s["dir"] / f"{stamp}_{s['count']}.jpg"), frame)
            msg = f"Sample {s['count']}/{config.SAMPLES_PER_PERSON} captured."
        return True, "ok", msg, self._state(session_id)

    def finish(self, session_id):
        """Persist metadata + embeddings. Returns (ok, message, info)."""
        with self.lock:
            s = self.sessions.get(session_id)
            if not s or not s["embeddings"]:
                return False, "Registration could not be completed. Please try again.", None
            if s["count"] < config.SAMPLES_PER_PERSON:
                return False, f"Only {s['count']}/{config.SAMPLES_PER_PERSON} samples captured.", None
            self.sessions.pop(session_id)
        meta = {"name": s["name"], "details": s["role"],
                "embedding_model": self.store.model_id,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        (s["dir"] / "metadata.json").write_text(json.dumps(meta, indent=4))
        if self.storage is None:
            self.store.add(s["embeddings"], s["name"], s["role"])
        else:
            try:
                self._add_and_upload(s)
            except Exception:
                with self.lock:  # keep the captured samples so "finish" can simply be retried
                    s["touched"] = time.time()
                    self.sessions[session_id] = s
                raise
        log.info("Registered a person with %d samples", s["count"])  # no personal data in logs
        return True, "Registration complete.", {"name": s["name"], "role": s["role"],
                                                "samples": s["count"]}

    def _add_and_upload(self, s):
        """S3 is the permanent store, so registration only counts once EVERYTHING is in S3.
        Order: photos -> metadata.json -> embeddings. If the embeddings upload fails the in-memory
        store is rolled back, so what is recognised always matches what S3 holds. Raises
        S3StorageError."""
        with self._persist_lock:  # two finishes must not upload their embeddings out of order
            self.storage.upload_person(s["dir"])
            before = self.store.snapshot()  # add() replaces these arrays, never edits them in place
            self.store.add(s["embeddings"], s["name"], s["role"])
            try:
                self.storage.upload_embeddings(self.store.path)
            except Exception:
                with self.store.lock:
                    self.store.embeddings, self.store.names, self.store.roles = before
                raise

    def _state(self, session_id):
        s = self.sessions.get(session_id)
        return {"count": s["count"] if s else 0, "target": config.SAMPLES_PER_PERSON}
