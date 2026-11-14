"""Central configuration. Every value can be overridden with an FR_* environment variable.

Directory layout (all paths are portable; nothing is hard-coded to one machine):

  models/            IMMUTABLE pretrained ONNX files (downloaded by scripts/download_models.py).
                     In Docker: /app/models, baked into the image, read-only in practice.
  data/              WRITABLE scratch copy of the registration data. In Docker: /data (plain container
                     storage, NOT a volume). With FR_S3_BUCKET set, Amazon S3 is the permanent source of
                     truth: data/ is rebuilt from S3 at every start and every registration is uploaded.
    known_faces/<Name>/   registration photos + metadata.json
    embeddings/           embeddings.npz  (vectors + names + roles + model id)
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _path(env, default):
    return Path(os.environ.get(env, default)).expanduser()


def _float(env, default):
    return float(os.environ.get(env, default))


def _int(env, default):
    return int(os.environ.get(env, default))


# ----------------------------------------------------------------- paths
MODELS_DIR = _path("FR_MODELS_DIR", BASE_DIR / "models")
DATA_DIR = _path("FR_DATA_DIR", BASE_DIR / "data")
KNOWN_FACES_DIR = _path("FR_KNOWN_FACES_DIR", DATA_DIR / "known_faces")
EMBEDDINGS_DIR = _path("FR_EMBEDDINGS_DIR", DATA_DIR / "embeddings")
# Where the OLD dlib version kept its .npy files. Only inspected (never read as embeddings,
# never modified or deleted) so we can tell the user that re-registration is needed.
LEGACY_DIR = _path("FR_LEGACY_DIR", DATA_DIR / "models")

# ----------------------------------------------------------------- Amazon S3 (permanent storage)
# Empty FR_S3_BUCKET = S3 disabled (local development / tests: nothing ever talks to AWS).
# Credentials are NEVER configured here: boto3 uses the EC2 instance role (FaceRecognitionEC2Role).
S3_BUCKET = os.environ.get("FR_S3_BUCKET", "").strip()
S3_REGION = os.environ.get("FR_S3_REGION", "ap-south-1").strip()
S3_KNOWN_FACES_PREFIX = "known_faces/"   # known_faces/<Name>/<photos>.jpg + metadata.json
S3_EMBEDDINGS_PREFIX = "embeddings/"     # embeddings/embeddings.npz

# ----------------------------------------------------------------- models (see scripts/download_models.py)
DETECTOR_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
# Stored with every embedding. Embeddings are only ever compared with embeddings carrying the
# SAME id - two models can both output 128 numbers and still be completely incompatible.
EMBEDDING_MODEL_ID = "sface_2021dec"
EMBEDDING_DIM = 128

# ----------------------------------------------------------------- detection (YuNet)
# DETECTION confidence: "how sure is the detector that this is a face". NOT identity confidence.
DETECT_SCORE_THRESHOLD = _float("FR_DETECT_SCORE", 0.9)   # OpenCV Zoo YuNet default
DETECT_NMS_THRESHOLD = _float("FR_DETECT_NMS", 0.3)       # OpenCV Zoo YuNet default
DETECT_TOP_K = _int("FR_DETECT_TOP_K", 5000)              # OpenCV Zoo YuNet default
MIN_FACE_SIZE = _int("FR_MIN_FACE_SIZE", 40)              # px; registration rejects smaller faces

# ----------------------------------------------------------------- recognition (SFace)
# RECOGNITION similarity: cosine similarity between two L2-normalised 128-d SFace embeddings
# (1.0 = identical direction, 0 = unrelated). Two faces are "the same person" when
# similarity >= threshold. 0.363 is the cosine threshold documented in the OpenCV Zoo SFace
# demo. It is a starting point, NOT a calibrated value for your cameras/people:
#   * raise it (e.g. 0.45) -> fewer false accepts, more "Unknown" for real people
#   * lower it (e.g. 0.30) -> fewer false rejects, more risk of mistaking one person for another
# The API field "distance" is 1 - similarity so existing UI text keeps working.
COSINE_THRESHOLD = _float("FR_COSINE_THRESHOLD", 0.60)

# ----------------------------------------------------------------- registration
SAMPLES_PER_PERSON = _int("FR_SAMPLES", 20)
SESSION_TTL_SECONDS = _int("FR_SESSION_TTL", 1800)  # unfinished registrations are discarded

# ----------------------------------------------------------------- request limits
MAX_IMAGE_SIDE = _int("FR_MAX_IMAGE_SIDE", 800)     # frames are downscaled to this before detection
MAX_UPLOAD_MB = _int("FR_MAX_UPLOAD_MB", 8)
MAX_IMAGE_PIXELS = _int("FR_MAX_IMAGE_PIXELS", 25_000_000)  # decompression-bomb guard
# OpenCV refuses to decode larger images; must be set before cv2 is first used.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(MAX_IMAGE_PIXELS))

# ----------------------------------------------------------------- Flask
HOST = os.environ.get("FR_HOST", "0.0.0.0")
PORT = _int("FR_PORT", 5000)
DEBUG = os.environ.get("FR_DEBUG", "0") == "1"
