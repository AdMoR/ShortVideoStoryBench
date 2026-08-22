---
name: mock-video
description: Generate a video from a text prompt using the local video generation service. Use this whenever you are asked to produce, generate, or create a video.
---

# Generating a video

A video generation service is running locally. Its base URL is in the environment
variable `MOCK_VIDEO_URL`.

## The one command you need

This generates the video and writes it to a file. It blocks until the video is
ready — that is the intended way to use it.

```bash
curl -sS -X POST "$MOCK_VIDEO_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "PUT THE FULL BRIEF HERE"}' \
  -o video.mp4
```

Generation can take several minutes. **Let the command finish.** Do not add a
timeout, and do not interrupt it to check on it — there is nothing to check.

Then confirm the file is real and hand it in:

```bash
ls -l video.mp4
```

If the file exists and is not empty, call `submit_video` with its path. You are done.

## If the blocking call fails

The service returns HTTP 500 on a failed generation, and `curl -o` will have
written the error JSON into `video.mp4`. Check for that:

```bash
head -c 100 video.mp4
```

If it looks like JSON rather than binary, the generation failed — run the same
command again.

## Optional: the asynchronous route

Only use this if the blocking endpoint is unavailable. Submit, then wait for the
result **inside a single command**, so the whole wait costs one step:

```bash
JOB=$(curl -sS -X POST "$MOCK_VIDEO_URL/jobs" \
        -H 'Content-Type: application/json' \
        -d '{"prompt": "PUT THE FULL BRIEF HERE"}' \
      | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)

for i in $(seq 1 120); do
  if curl -sS "$MOCK_VIDEO_URL/jobs/$JOB" | grep -q '"status":"completed"'; then
    curl -sS "$MOCK_VIDEO_URL/jobs/$JOB/video" -o video.mp4
    break
  fi
  sleep 5
done
```

Do not split that loop across several commands — one command per poll would cost
a full reasoning step each time and fill your context with repeated status output.
