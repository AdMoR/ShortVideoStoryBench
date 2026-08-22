"""
Video frame extraction for the judge.

The judge is a vision LLM that cannot watch video directly, so we sample N
evenly-spaced frames (in temporal order) and send them as JPEG images.
"""

import io
import logging
from pathlib import Path
from typing import List

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, n: int = 8, max_width: int = 768) -> List[bytes]:
    """
    Extract n evenly-spaced frames from a video as JPEG bytes.

    Args:
        video_path: Path to the video file.
        n: Number of frames to extract (default 8).
        max_width: Frames are downscaled to this width to keep payloads small.

    Returns:
        List of JPEG bytes, length <= n. Empty list if the video cannot be opened.
    """
    path = Path(video_path)
    if not path.exists():
        logger.warning(f"Video not found: {video_path}")
        return []

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    if n == 1:
        indices = [total_frames // 2]
    else:
        indices = [int(i * (total_frames - 1) / (n - 1)) for i in range(n)]

    frames: List[bytes] = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb, mode="RGB")
            if img.width > max_width:
                scale = max_width / img.width
                img = img.resize((max_width, int(img.height * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            frames.append(buf.getvalue())
    finally:
        cap.release()

    logger.info(f"Extracted {len(frames)}/{n} frames from {video_path}")
    return frames


def video_frame_count(video_path: str) -> int:
    """
    Number of frames OpenCV reports for a video, or 0 if it cannot be read.

    Generators use this to check a produced file before handing it on. It
    deliberately asks the same question `extract_frames` asks — a file that
    answers 0 here is one the judge would refuse, so it is better rejected at
    generation time where the error can name the generator.
    """
    path = Path(video_path)
    if not path.exists():
        return 0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0:
            return count
        # Some containers report no frame count; fall back to reading one frame.
        ok, _ = cap.read()
        return 1 if ok else 0
    finally:
        cap.release()


def load_image(image_path: str, max_width: int = 1024) -> bytes:
    """Load a still image (reference frame, etc.) and return JPEG bytes."""
    path = Path(image_path)
    if not path.exists():
        logger.warning(f"Image not found: {image_path}")
        return b""
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        scale = max_width / img.width
        img = img.resize((max_width, int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
