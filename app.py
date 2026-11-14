"""Flask entry point.  python app.py   or   gunicorn 'app:create_app()'"""
import logging
import time

from flask import Flask, jsonify, render_template, request

import config
from backend.services.face_recognition import EmbeddingStore, Recognizer
from backend.services.registration import Registrar, valid_session_id
from backend.services.s3_storage import S3Storage, S3StorageError
from backend.utils.image_utils import decode_data_url, downscale

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


def create_app(engine=None, store=None, known_dir=None, storage=None):
    """Models are loaded ONCE here. Tests inject a fake engine/store instead.

    Permanent storage: when the app builds its own store and FR_S3_BUCKET is set (or a `storage`
    is injected) everything registered so far is restored from S3 BEFORE the store loads. If that
    fails the app refuses to start. An injected `store` is used as-is: nothing is restored and
    S3 is only touched if `storage` is injected too, so tests never need AWS credentials."""
    app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

    engine_error = None
    if engine is None:
        try:
            from backend.services.face_engine import FaceEngine
            engine = FaceEngine()
        except Exception as exc:  # missing/corrupt model files -> app still starts, reports it
            engine_error = str(exc)
            log.error("Recognition engine failed to load: %s", exc)

    if store is None:
        if storage is None and config.S3_BUCKET:
            storage = S3Storage(known_dir=known_dir)
        if storage is not None:
            try:
                storage.restore()  # downloads photos, metadata and embeddings; empty bucket is fine
            except S3StorageError as exc:
                log.critical("STARTUP ABORTED: cannot restore registration data from S3: %s", exc)
                raise
        store = EmbeddingStore()
    ready = engine_error is None
    recognizer = Recognizer(engine, store) if ready else None
    registrar = Registrar(engine, store, known_dir, storage) if ready else None
    started = time.time()

    def fail(msg, code):
        return jsonify({"success": False, "error": msg}), code

    def json_body():
        body = request.get_json(silent=True)
        return body if isinstance(body, dict) else {}

    # ---------- pages ----------
    @app.get("/")
    def index():
        return render_template("index.html", page="home")

    @app.get("/register")
    def register_page():
        return render_template("register.html", page="register", samples=config.SAMPLES_PER_PERSON)

    @app.get("/recognize")
    def recognize_page():
        return render_template("recognize.html", page="recognize")

    # ---------- API ----------
    @app.get("/health")
    def health():
        # Readiness, not just "the process is up": the face engine (YuNet + SFace models) must be
        # loaded. 503 lets the Docker HEALTHCHECK, the Ansible check and CI treat a broken deploy
        # as failed. The reason is only written to the log (it can contain file paths).
        if not ready:
            return jsonify({"status": "unhealthy", "engine_ready": False}), 503
        return jsonify({"status": "healthy"})

    @app.get("/api/status")
    def status():
        return jsonify({"engine_ready": ready,
                        "registered_users": len(store),
                        "model": store.model_id,
                        "cosine_threshold": config.COSINE_THRESHOLD,
                        "tolerance": config.COSINE_THRESHOLD,  # kept for older clients
                        "samples_required": config.SAMPLES_PER_PERSON,
                        "legacy_data_detected": store.legacy_detected(),
                        "incompatible_store": store.incompatible,
                        "uptime_seconds": int(time.time() - started)})

    @app.post("/api/recognize")
    def recognize():
        if not ready:
            return fail("Recognition engine is not available.", 503)
        frame = decode_data_url(json_body().get("image"))
        if frame is None:
            return fail("Invalid or missing image.", 400)
        frame, scale = downscale(frame)
        try:
            faces = recognizer.recognize(frame)
        except Exception:
            log.exception("Recognition failed")
            return fail("Recognition failed. Please try again.", 500)
        for f in faces:  # map boxes back to the ORIGINAL image size
            f["box"] = [int(round(v / scale)) for v in f["box"]]
        h, w = frame.shape[:2]
        return jsonify({"success": True, "faces": faces,
                        "frame": {"width": int(round(w / scale)), "height": int(round(h / scale))}})

    @app.post("/api/register")
    def register():
        if not ready:
            return fail("Recognition engine is not available.", 503)
        body = json_body()
        name, role, session_id = body.get("name"), body.get("role"), body.get("session_id")
        err = registrar.validate_person(name, role)
        if err:
            return fail(err, 400)
        if not valid_session_id(session_id):
            return fail("Missing or invalid session_id.", 400)

        if body.get("finish") is True:
            try:
                ok, msg, info = registrar.finish(session_id)
            except S3StorageError as exc:  # nothing is confirmed unless it reached S3
                log.error("Registration NOT saved - S3 upload failed: %s", exc)
                return fail("Registration could not be saved to permanent storage (Amazon S3). "
                            "Nothing was registered - please try again.", 503)
            if not ok:
                return fail(msg, 400)
            return jsonify({"success": True, "message": msg, **info, "registered_users": len(store)})

        frame = decode_data_url(body.get("image"))
        if frame is not None:
            frame, _ = downscale(frame)
        try:
            ok, code, msg, state = registrar.add_sample(session_id, name, role, frame)
        except Exception:
            log.exception("Registration failed")
            return fail("Registration could not be completed. Please try again.", 500)
        if code == "invalid_person":
            return fail(msg, 400)
        # A rejected sample is a normal outcome (200) - the UI just shows the hint.
        return jsonify({"success": True, "accepted": ok, "code": code, "message": msg, **(state or {})})

    @app.errorhandler(413)
    def too_big(_):
        return fail("Image is too large.", 413)

    @app.errorhandler(404)
    def not_found(_):
        return fail("Not found.", 404)

    return app


if __name__ == "__main__":
    create_app().run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
