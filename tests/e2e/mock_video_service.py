"""
A stand-in video generation backend for the end-to-end tier.

The e2e tests run the real pi agent against the real model — only the thing that
would take twenty minutes is faked. This service has the shape of a real backend
(a blocking endpoint and an async submit/poll pair) so the agent has to behave as
it would in production, but it answers in milliseconds unless told otherwise.

Standalone:

    python tests/e2e/mock_video_service.py --port 8099

Knobs (constructor args, or the matching env vars when standalone):

    MOCK_VIDEO_DELAY       seconds each generation takes (default 0)
    MOCK_VIDEO_FAIL_FIRST  fail the first generation, to see whether the agent retries
"""

import argparse
import json
import os
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def render_video(path: Path, seed_text: str, n_frames: int = 24) -> bytes:
    """A real, decodable mp4 — the judge has to be able to score it."""
    import cv2

    from video_eval_bench.generator.mock_generator import numpy_full_frame

    size = (320, 180)
    tone = sum(seed_text.encode()) % 200
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, size)
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path}")
    for i in range(n_frames):
        writer.write(numpy_full_frame((tone, 120, 200 - tone), size, i))
    writer.release()
    return path.read_bytes()


class MockVideoService:
    """Threaded HTTP server; start() returns the base URL."""

    def __init__(self, delay: float = 0.0, fail_first: bool = False):
        self.delay = delay
        self.fail_first = fail_first
        self.calls = []
        self.jobs = {}
        self._tmp = tempfile.TemporaryDirectory(prefix="mock_video_")
        self._dir = Path(self._tmp.name)
        self._server = None
        self._thread = None
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, port: int = 0) -> str:
        self._server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, bound = self._server.server_address[:2]
        return f"http://{host}:{bound}"

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        self._tmp.cleanup()

    # ── generation ────────────────────────────────────────────────────────────

    def _should_fail(self) -> bool:
        with self._lock:
            if self.fail_first and len(self.calls) == 1:
                return True
        return False

    def generate(self, prompt: str) -> bytes:
        """Blocking generation — the path the director prompt tells the agent to prefer."""
        with self._lock:
            self.calls.append(prompt)
        if self._should_fail():
            raise RuntimeError("generation failed (injected)")
        time.sleep(self.delay)
        return render_video(self._dir / f"{uuid.uuid4().hex}.mp4", prompt)

    def submit_job(self, prompt: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self.calls.append(prompt)
            self.jobs[job_id] = {"status": "running", "started": time.monotonic(), "data": None}
        threading.Thread(target=self._work, args=(job_id, prompt), daemon=True).start()
        return job_id

    def _work(self, job_id: str, prompt: str) -> None:
        time.sleep(self.delay)
        data = render_video(self._dir / f"{job_id}.mp4", prompt)
        with self._lock:
            self.jobs[job_id] = {"status": "completed", "data": data}

    def job_status(self, job_id: str) -> dict:
        with self._lock:
            return dict(self.jobs.get(job_id) or {})


def _make_handler(service: "MockVideoService"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the pytest output readable
            pass

        # ── helpers ───────────────────────────────────────────────────────────

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _video(self, data: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        # ── routes ────────────────────────────────────────────────────────────

        def do_GET(self):
            if self.path in ("/health", "/"):
                return self._json(200, {"ok": True})
            parts = self.path.strip("/").split("/")
            if parts[0] == "jobs" and len(parts) >= 2:
                job = service.job_status(parts[1])
                if not job:
                    return self._json(404, {"error": "no such job"})
                if len(parts) == 3 and parts[2] == "video":
                    if job["status"] != "completed":
                        return self._json(409, {"error": "not ready", "status": job["status"]})
                    return self._video(job["data"])
                return self._json(
                    200,
                    {
                        "job_id": parts[1],
                        "status": job["status"],
                        "video_url": f"/jobs/{parts[1]}/video",
                    },
                )
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            body = self._body()
            prompt = str(body.get("prompt", ""))
            if self.path.rstrip("/") == "/generate":
                try:
                    return self._video(service.generate(prompt))
                except RuntimeError as exc:
                    return self._json(500, {"error": str(exc)})
            if self.path.rstrip("/") == "/jobs":
                return self._json(202, {"job_id": service.submit_job(prompt)})
            return self._json(404, {"error": "not found"})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock video generation backend")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()

    service = MockVideoService(
        delay=float(os.environ.get("MOCK_VIDEO_DELAY", "0")),
        fail_first=os.environ.get("MOCK_VIDEO_FAIL_FIRST", "").lower() in ("1", "true", "yes"),
    )
    url = service.start(args.port)
    print(f"mock video service on {url}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
