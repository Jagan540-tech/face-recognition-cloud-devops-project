"""Embedding store + matching.

The store is ONE file (embeddings.npz) holding the vectors, names, roles AND the id of the model
that produced them. On load the id must equal the running model's id; otherwise the file is
treated as incompatible: it is NOT used, NOT modified and NOT deleted (it is backed up before a
new store replaces it), and /api/status reports it so the user knows to re-register.
"""
import logging
import os
import tempfile
import threading
import time

import numpy as np

import config
from backend.services.face_engine import normalize

log = logging.getLogger(__name__)

STORE_FILE = "embeddings.npz"
LEGACY_FILES = ("encodings.npy", "names.npy", "details.npy")


class EmbeddingStore:
    def __init__(self, directory=None, model_id=None, dim=None, legacy_dir=None):
        self.dir = directory or config.EMBEDDINGS_DIR
        self.model_id = model_id or config.EMBEDDING_MODEL_ID
        self.dim = dim or config.EMBEDDING_DIM
        self.legacy_dir = legacy_dir if legacy_dir is not None else config.LEGACY_DIR
        self.lock = threading.Lock()
        self._empty()
        self.incompatible = None   # description of an unusable store file, if one exists
        self.load()

    # ---------------------------------------------------------------- state
    def _empty(self):
        self.embeddings = np.empty((0, self.dim), dtype=np.float32)
        self.names = np.array([], dtype=str)
        self.roles = np.array([], dtype=str)

    @property
    def path(self):
        return self.dir / STORE_FILE

    def legacy_detected(self):
        """True if the old dlib .npy files are present (they are never loaded)."""
        d = self.legacy_dir
        return bool(d) and all((d / f).exists() for f in LEGACY_FILES)

    def load(self):
        self._empty()
        self.incompatible = None
        if self.legacy_detected():
            log.warning("Legacy dlib encodings found in %s. They are NOT compatible with the "
                        "current model (%s) and are left untouched. Register people again.",
                        self.legacy_dir, self.model_id)
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as z:
                model_id = str(z["model_id"])
                dim = int(z["dim"])
                emb, names, roles = z["embeddings"], z["names"], z["roles"]
        except Exception as exc:  # corrupt / unreadable
            self.incompatible = f"unreadable store ({exc.__class__.__name__})"
            log.error("Cannot read %s: %s. File left untouched.", self.path, exc)
            return
        if model_id != self.model_id or dim != self.dim:
            self.incompatible = f"store was built with model '{model_id}' (dim {dim}), " \
                                f"running model is '{self.model_id}' (dim {self.dim})"
            log.error("Incompatible embedding store: %s. File left untouched.", self.incompatible)
            return
        if not (len(emb) == len(names) == len(roles)) or (emb.size and emb.shape[1] != self.dim):
            self.incompatible = "inconsistent array lengths"
            log.error("Embedding store %s is inconsistent. File left untouched.", self.path)
            return
        self.embeddings = emb.astype(np.float32).reshape(-1, self.dim)
        self.names, self.roles = names.astype(str), roles.astype(str)

    def _save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.incompatible and self.path.exists():  # never overwrite data we could not read
            backup = self.path.with_name(f"{self.path.stem}.incompatible-{int(time.time())}.npz")
            os.replace(self.path, backup)
            log.warning("Moved unusable store to %s", backup)
            self.incompatible = None
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp, "wb") as fh:  # atomic: a crash never leaves a half-written store
                np.savez(fh, embeddings=self.embeddings, names=self.names, roles=self.roles,
                         model_id=np.array(self.model_id), dim=np.array(self.dim))
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ---------------------------------------------------------------- operations
    def add(self, vectors, name, role):
        """Store embeddings for one person, replacing any earlier registration of that name."""
        vecs = np.vstack([normalize(v) for v in vectors]).astype(np.float32)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"embedding dim {vecs.shape[1]} != {self.dim}")
        with self.lock:
            keep = self.names != name if len(self.names) else np.zeros(0, dtype=bool)
            self.embeddings = np.vstack([self.embeddings[keep], vecs])
            self.names = np.concatenate([self.names[keep], np.array([name] * len(vecs))])
            self.roles = np.concatenate([self.roles[keep], np.array([role] * len(vecs))])
            self._save()

    def snapshot(self):
        with self.lock:
            return self.embeddings, self.names, self.roles

    def people(self):
        with self.lock:
            return sorted(set(self.names.tolist()))

    def __len__(self):
        return len(self.people())


class Recognizer:
    def __init__(self, engine, store):
        self.engine, self.store = engine, store

    def recognize(self, frame_bgr):
        """One result per detected face. box = [x, y, w, h] in frame pixels.
        similarity = cosine (higher = more alike); distance = 1 - similarity."""
        faces = self.engine.detect(frame_bgr)
        emb, names, roles = self.store.snapshot()
        results = []
        for face in faces:
            r = {"recognized": False, "name": "Unknown", "role": None, "distance": None,
                 "similarity": None, "detection_score": round(face.score, 4),
                 "box": [int(v) for v in face.box]}
            if len(emb):
                query = self.engine.embed(frame_bgr, face)
                sims = emb @ query  # both sides are unit-length -> cosine similarity
                best = int(np.argmax(sims))
                sim = float(sims[best])
                r["similarity"] = round(sim, 4)
                r["distance"] = round(1.0 - sim, 4)
                if sim >= config.COSINE_THRESHOLD:
                    r.update(recognized=True, name=str(names[best]), role=str(roles[best]))
            results.append(r)
        return results
