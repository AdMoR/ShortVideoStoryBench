from typing import Awaitable, Callable, Dict, List, Optional
import logging
import numpy as np
from video_eval_bench.dataset.seed import Seed
from pathlib import Path
import cv2
import hashlib


logger = logging.getLogger(__name__)


def numpy_full_frame(color, size, i: int):

    frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    # A moving white square so the frames are not identical (motion signal).
    x = (i * 20) % max(1, size[0] - 40)
    frame[20:60, x : x + 40] = 255
    return frame


class MockGenerator:
    """
    Offline generator: writes a tiny synthetic video (solid color frames via
    OpenCV) so the full bench pipeline can run without a video model.
    """

    def __init__(self, n_frames: int = 16, fps: int = 8, size=(320, 180)):
        self.n_frames = n_frames
        self.fps = fps
        self.size = size
        self.calls: List[str] = []

    async def __call__(self, seed: Seed, output_dir: Path) -> str:

        self.calls.append(seed.seed_id)
        # Deterministic color per seed so different seeds look different.
        h = int(hashlib.md5(seed.seed_id.encode()).hexdigest()[:6], 16)
        color = ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)  # BGR

        out_path = output_dir / f"{seed.seed_id}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.size,
        )
        if not writer.isOpened():
            # Fallback for OpenCV builds without an mp4 encoder.
            out_path = output_dir / f"{seed.seed_id}.avi"
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self.fps,
                self.size,
            )
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {out_path}")
        for i in range(self.n_frames):
            frame = numpy_full_frame(color, self.size, i)
            writer.write(frame)
        writer.release()
        logger.info(f"[MockGenerator] wrote {out_path}")
        return str(out_path)
