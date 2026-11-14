"""Helpers for turning browser-sent images into OpenCV frames."""
import base64
import binascii

import cv2
import numpy as np

import config


def decode_data_url(data):
    """Decode 'data:image/jpeg;base64,...' (or raw base64) into a BGR ndarray.
    Returns None if the data is not a decodable image."""
    if not data or not isinstance(data, str):
        return None
    if data.lstrip().startswith("data:"):
        if "," not in data:
            return None
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not raw:
        return None
    try:
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if img is None or img.ndim != 3 or img.shape[0] < 16 or img.shape[1] < 16:
        return None
    return img


def downscale(frame):
    """Shrink so the longest side <= MAX_IMAGE_SIDE. Returns (frame, scale), scale <= 1."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= config.MAX_IMAGE_SIDE:
        return frame, 1.0
    s = config.MAX_IMAGE_SIDE / longest
    return cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA), s
