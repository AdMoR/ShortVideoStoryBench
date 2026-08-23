"""
A stand-in for the `pi` binary, for offline tests.

Emits pi's `--mode json` NDJSON on stdout and behaves according to FAKE_PI_MODE,
so every branch of the generator's process lifecycle can be exercised without a
model, a network, or Node. It reads VEB_OUTPUT_PATH / VEB_WORKSPACE from the
environment exactly as the real `bench_tools.ts` extension does.

Modes:

    submit             produce a video, submit it, exit 0            (happy path)
    no_submit          work, then finish without submitting
    fail               write to stderr and exit 3
    submit_then_fail   submit a good video, then exit 3
    empty_file         submit a 0-byte file
    file_only          write the video but never emit the submit event
    resubmit           submit a video, then submit a better one over it
    linger             submit, then keep running past the submission
    silent             go quiet past the heartbeat interval, then submit
    stderr_flood       write megabytes to stderr, then submit
    hang               never produce anything, never exit
"""

import json
import os
import sys
import time


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def write_video(path: str, n_frames: int = 12) -> int:
    """A real, decodable video — the judge's frame extractor must be able to open it."""
    import cv2

    from video_eval_bench.generator.mock_generator import numpy_full_frame

    size = (320, 180)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 8, size)
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path}")
    for i in range(n_frames):
        writer.write(numpy_full_frame((40, 90, 160), size, i))
    writer.release()
    return os.path.getsize(path)


def start() -> None:
    emit({"type": "session", "version": 3, "id": "fake", "cwd": os.getcwd()})
    emit({"type": "agent_start"})
    emit({"type": "turn_start"})


def work(tool: str = "bash") -> None:
    emit({"type": "tool_execution_start", "toolCallId": "t1", "toolName": tool, "args": {}})
    emit(
        {
            "type": "tool_execution_end",
            "toolCallId": "t1",
            "toolName": tool,
            "result": {"content": [{"type": "text", "text": "ok"}]},
            "isError": False,
        }
    )
    emit(
        {
            "type": "message_update",
            # Real pi reports totalTokens alongside the breakdown, and emits an
            # all-zeros block on message_start — both shapes matter downstream.
            "usage": {"input": 1200, "output": 340, "totalTokens": 1540},
            "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": "..."},
        }
    )


def submit(destination: str, size: int, source: str) -> None:
    emit(
        {
            "type": "tool_execution_start",
            "toolCallId": "t2",
            "toolName": "submit_video",
            "args": {"path": source},
        }
    )
    emit(
        {
            "type": "tool_execution_end",
            "toolCallId": "t2",
            "toolName": "submit_video",
            "result": {
                "content": [{"type": "text", "text": f"Submitted {source}"}],
                "details": {
                    "kind": "submit_video",
                    "path": destination,
                    "source": source,
                    "bytes": size,
                    "notes": "fake",
                },
            },
            "isError": False,
        }
    )


def finish() -> None:
    emit({"type": "turn_end", "message": {"role": "assistant", "content": []}, "toolResults": []})
    emit({"type": "agent_end", "messages": []})


def main() -> int:
    mode = os.environ.get("FAKE_PI_MODE", "submit")
    destination = os.environ["VEB_OUTPUT_PATH"]
    workspace = os.environ.get("VEB_WORKSPACE", os.getcwd())
    produced = os.path.join(workspace, "out.mp4")

    if mode == "hang":
        start()
        work()
        time.sleep(3600)
        return 0

    if mode == "fail":
        start()
        sys.stderr.write("fake pi exploded\n" + "detail line\n" * 20)
        sys.stderr.flush()
        return 3

    if mode == "no_submit":
        start()
        work()
        finish()
        return 0

    if mode == "file_only":
        start()
        work()
        write_video(destination)
        finish()
        return 0

    if mode == "empty_file":
        start()
        work()
        open(produced, "wb").close()
        open(destination, "wb").close()
        submit(destination, 0, produced)
        finish()
        return 0

    if mode == "stderr_flood":
        start()
        # Far more than a pipe buffer: an implementation that drains stdout and
        # stderr sequentially deadlocks here instead of finishing.
        chunk = ("noise " * 200 + "\n") * 100
        for _ in range(40):
            sys.stderr.write(chunk)
        sys.stderr.flush()
        work()

    if mode == "silent":
        start()
        emit({"type": "tool_execution_start", "toolCallId": "t9", "toolName": "bash", "args": {}})
        time.sleep(float(os.environ.get("FAKE_PI_SILENCE", "2.5")))

    if mode not in {"stderr_flood", "silent"}:
        start()
        work()

    size = write_video(produced)
    import shutil

    shutil.copyfile(produced, destination)
    submit(destination, size, produced)

    if mode == "resubmit":
        # Bank a first result, then improve it and submit again. The second video
        # is visibly longer, so a harness that kept the first one is detectable.
        second = os.path.join(workspace, "better.mp4")
        second_size = write_video(second, n_frames=24)
        shutil.copyfile(second, destination)
        submit(destination, second_size, second)
        finish()
        return 0

    if mode == "linger":
        finish()
        time.sleep(float(os.environ.get("FAKE_PI_LINGER", "60")))
        return 0

    if mode == "submit_then_fail":
        sys.stderr.write("cleanup failed after a good submission\n")
        sys.stderr.flush()
        return 3

    finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
