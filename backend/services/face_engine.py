"""Face detection + embedding on OpenCV's built-in DNN module. No TensorFlow, no PyTorch,
no onnxruntime, no dlib: the only ML runtime is opencv-python-headless itself.

  * Detector   : YuNet   (face_detection_yunet_2023mar.onnx)   - MIT,        ~0.23 MB
  * Recognizer : SFace   (face_recognition_sface_2021dec.onnx)  - Apache-2.0, ~37 MB
Both come from the OpenCV Zoo; see scripts/download_models.py for pinned hashes.

Colour handling: OpenCV frames are BGR and both models expect BGR. Nothing here converts to RGB
(the old dlib/MTCNN code needed RGB; feeding RGB to these models silently hurts accuracy).
"""
import threading
from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class Face:
    box: tuple          # (x, y, w, h) ints, clipped to the frame, w,h > 0
    score: float        # DETECTION confidence from YuNet (not identity confidence)
    landmarks: np.ndarray  # 5 points (right eye, left eye, nose, right mouth, left mouth)
    raw: np.ndarray     # the 15-float YuNet row; SFace's alignCrop needs it as-is


class FaceEngine:
    model_id = config.EMBEDDING_MODEL_ID
    dim = config.EMBEDDING_DIM

    def __init__(self, detector_path=None, recognizer_path=None):
        detector_path = detector_path or config.DETECTOR_MODEL
        recognizer_path = recognizer_path or config.RECOGNIZER_MODEL
        for p in (detector_path, recognizer_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Model file not found: {p}. Run: python scripts/download_models.py")
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320),
            config.DETECT_SCORE_THRESHOLD, config.DETECT_NMS_THRESHOLD, config.DETECT_TOP_K)
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        # YuNet keeps its input size as state (setInputSize) and OpenCV DNN nets are not
        # thread-safe, so inference is serialised. It is CPU-bound anyway.
        self._lock = threading.Lock()

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        with self._lock:
            self._detector.setInputSize((w, h))
            _, rows = self._detector.detect(frame_bgr)
        faces = []
        if rows is None:
            return faces
        for row in rows:
            score = float(row[14])
            if score < config.DETECT_SCORE_THRESHOLD:
                continue
            x, y, bw, bh = row[:4]
            x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
            x2, y2 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            faces.append(Face((x1, y1, x2 - x1, y2 - y1), score,
                              row[4:14].reshape(5, 2).copy(), row.astype(np.float32)))
        return faces

    def embed(self, frame_bgr, face):
        """Aligned 112x112 crop (via the 5 landmarks) -> L2-normalised float32 vector (128,)."""
        with self._lock:
            aligned = self._recognizer.alignCrop(frame_bgr, face.raw)
            feat = self._recognizer.feature(aligned)
        return normalize(np.asarray(feat, dtype=np.float32).reshape(-1))


def normalize(vec):
    """L2-normalise so that a dot product IS the cosine similarity."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    if not np.isfinite(n) or n < 1e-8:
        raise ValueError("degenerate embedding")
    return vec / n
