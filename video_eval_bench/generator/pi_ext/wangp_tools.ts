/**
 * Video generation tool: MiniMax H3 via the WanGP REST server.
 *
 * Wraps the four-step dance (upload refs → POST /jobs/raw → poll → download)
 * behind one blocking `generate_video` call, so the agent spends a single tool
 * call on a generation that takes minutes. That is the whole point: polling from
 * the agent loop costs a model call per check, floods the context with repeated
 * status output, and pushes a long run into auto-compaction. Waiting inside the
 * tool costs one turn no matter how long it takes.
 *
 * Two modes, matching what H3 actually supports:
 *
 *   t2va   text only — no first/last frame needed. Sent as fl2va with
 *          image_prompt_type unset, which the server accepts as a
 *          "type the dialogue, no visual anchor" call.
 *   ref2va reference-driven — one or more local images conditioning identity
 *          or setting. Uploaded first, then referenced as "file:<id>".
 *
 * Field shapes and defaults follow virtual_streamer/video_generation/h3_client.py
 * (H3GenerationParams); the fixed-by-model values are not tunable knobs.
 */

import { createReadStream, statSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { isAbsolute, join, resolve, sep } from "node:path";
import { Type } from "typebox";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const SERVER = () => (process.env.WANGP_SERVER_URL ?? "http://gx10-cbc5:8082").replace(/\/$/, "");
const API_KEY = () => process.env.WANGP_API_KEY ?? "";
const WORKSPACE = () => process.env.VEB_WORKSPACE ?? process.cwd();

// Output is 24fps and the frame count sits on a 107 + 17n grid; off-grid values
// are rounded down by the server, so snap here and say what we did.
const FRAMES_MINIMUM = 107;
const FRAMES_STEPS = 17;
const FPS = 24;
const BLOCK_SIZE = 32;

const DEFAULT_NEGATIVE_PROMPT =
	"worst quality, inconsistent motion, blurry, jittery, distorted";

const POLL_INTERVAL_MS = 5000;

// A dropped connection must not discard a generation the server is still doing.
// The GPU keeps working, the job record stays valid, and a single failed poll
// throws away minutes of compute that are already paid for. Observed in
// runs/20260823-002540: four calls died on a bare `TypeError: fetch failed`
// while their jobs ran to completion server-side. Finished jobs are then
// evicted, so those videos were unrecoverable.
const POLL_RETRY_LIMIT = 12; // consecutive transient failures tolerated
const POLL_BACKOFF_MS = 2000; // grows linearly per consecutive failure
const POLL_BACKOFF_MAX_MS = 30_000;
const DOWNLOAD_RETRY_LIMIT = 5;

// A last-resort cap so a job the server wedges in "running" cannot hold the tool
// open forever. Generations of ~31 minutes are normal, so this is deliberately
// far above them; the agent's own budget is the usual stopping condition.
const MAX_WAIT_MS = Number(process.env.WANGP_MAX_WAIT_MS ?? 3 * 60 * 60 * 1000);

function headers(extra: Record<string, string> = {}): Record<string, string> {
	return API_KEY() ? { ...extra, "X-API-Key": API_KEY() } : extra;
}

/** Snap a duration in seconds onto the frame grid the server accepts. */
function framesFor(seconds: number): number {
	const wanted = Math.round(seconds * FPS);
	const steps = Math.max(0, Math.round((wanted - FRAMES_MINIMUM) / FRAMES_STEPS));
	return FRAMES_MINIMUM + steps * FRAMES_STEPS;
}

/** Both dimensions must be multiples of BLOCK_SIZE. */
function snapResolution(resolution: string): string {
	const match = /^(\d+)\s*x\s*(\d+)$/i.exec(resolution.trim());
	if (!match) return "832x480";
	const snap = (n: number) => Math.max(BLOCK_SIZE, Math.round(n / BLOCK_SIZE) * BLOCK_SIZE);
	return `${snap(Number(match[1]))}x${snap(Number(match[2]))}`;
}

function inWorkspace(path: string): string | null {
	const workspace = resolve(WORKSPACE());
	const candidate = isAbsolute(path) ? resolve(path) : resolve(workspace, path);
	if (candidate !== workspace && !candidate.startsWith(workspace + sep)) return null;
	return candidate;
}

function fail(why: string, extra: Record<string, unknown> = {}) {
	return {
		content: [{ type: "text" as const, text: `generate_video failed: ${why}` }],
		details: { kind: "generate_video_error", error: why, ...extra },
		isError: true,
	};
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** A failure that retrying cannot fix — the job is gone or the server refused it. */
class FatalJobError extends Error {}

/** An HTTP status the server itself says is worth retrying (5xx, 429). */
class TransientHttpError extends Error {}

/**
 * Connection-level failures deserve another attempt; a cancelled call does not.
 *
 * Node reports a dropped socket as a bare `TypeError: fetch failed` and puts the
 * real reason on `.cause`, so both are inspected. Retrying a refused connection
 * is intentional: the server is restarted between model loads, and the job
 * survives that.
 */
function isTransient(error: unknown, signal?: AbortSignal): boolean {
	if (signal?.aborted) return false;
	if (error instanceof FatalJobError) return false;
	if (error instanceof TransientHttpError) return true;
	if (error instanceof Error && error.name === "AbortError") return false;
	const text = `${error} ${(error as { cause?: unknown })?.cause ?? ""}`;
	return /fetch failed|socket hang up|terminated|other side closed|ECONNRESET|ECONNREFUSED|ETIMEDOUT|EPIPE|EAI_AGAIN|ENOTFOUND|UND_ERR/i.test(
		text,
	);
}

/** 5xx and 429 are the server asking for another try; anything else is final. */
function retryableStatus(status: number): boolean {
	return status === 429 || status >= 500;
}

/** POST /files/upload → file_id */
async function uploadReference(path: string, signal?: AbortSignal): Promise<string> {
	const stat = statSync(path);
	if (!stat.isFile()) throw new Error(`${path} is not a file`);
	const form = new FormData();
	// Node 22: a Blob from the file bytes keeps this dependency-free.
	const bytes = await new Response(createReadStream(path) as any).arrayBuffer();
	form.append("file", new Blob([bytes]), path.split(sep).pop() ?? "ref");

	const response = await fetch(`${SERVER()}/files/upload`, {
		method: "POST",
		headers: headers(),
		body: form,
		signal,
	});
	if (!response.ok) {
		throw new Error(`upload of ${path} failed: HTTP ${response.status}`);
	}
	const body: any = await response.json();
	if (!body?.file_id) throw new Error(`upload of ${path} returned no file_id`);
	return body.file_id as string;
}

export default function wangpTools(pi: ExtensionAPI) {
	pi.registerTool({
		name: "generate_video",
		label: "Generate Video",
		description:
			"Generate a video with synchronized audio from a text prompt using the MiniMax H3 " +
			"model. Blocks until the video is ready and saves it into the working directory, " +
			"which can take several minutes — that is normal, and it costs you only this one " +
			"call. Use mode 't2va' for text only, or 'ref2va' with image_refs to condition on " +
			"reference images.",
		promptSnippet: "Generate a video with audio from a text prompt (blocking)",
		promptGuidelines: [
			"Use generate_video to produce video; it is the only generator available and it blocks until finished.",
			"Do not poll or sleep after calling generate_video — it returns when the video is on disk.",
			"Write the full scene description into generate_video's prompt: shots, camera motion, and any spoken lines.",
		],
		parameters: Type.Object({
			prompt: Type.String({
				description:
					"Full production brief: shots with timing, explicit camera motion, and any dialogue.",
			}),
			mode: Type.Optional(
				StringEnum(["t2va", "ref2va"] as const, {
					description: "t2va = text only (default). ref2va = conditioned on image_refs.",
				}),
			),
			image_refs: Type.Optional(
				Type.Array(Type.String(), {
					description:
						"ref2va only: paths to reference images in the working directory.",
				}),
			),
			duration_seconds: Type.Optional(
				Type.Number({ description: "4-15 seconds (default 5). Snapped to the frame grid." }),
			),
			resolution: Type.Optional(
				Type.String({ description: "WxH, default 832x480. Snapped to a multiple of 32." }),
			),
			quality: Type.Optional(
				StringEnum(["turbo", "standard"] as const, {
					description:
						"turbo = 4-step, ~5x faster, and the variant the server preloads (default). " +
						"standard = ~20 steps, slower and better.",
				}),
			),
			seed: Type.Optional(Type.Number({ description: "-1 for random (default)." })),
		}),

		async execute(_toolCallId, params, signal, onUpdate) {
			const mode = params.mode ?? "t2va";
			const turbo = (params.quality ?? "turbo") === "turbo";
			const prompt = (params.prompt ?? "").trim();
			if (!prompt) return fail("prompt is empty");

			// ── references (ref2va) ──────────────────────────────────────────
			const refs: string[] = [];
			if (mode === "ref2va") {
				if (!params.image_refs?.length) {
					return fail("mode 'ref2va' needs at least one path in image_refs");
				}
				for (const raw of params.image_refs) {
					const path = inWorkspace(raw);
					if (!path) return fail(`reference ${raw} is outside the working directory`);
					try {
						onUpdate?.({ content: [{ type: "text", text: `uploading ${raw}…` }] });
						refs.push(`file:${await uploadReference(path, signal)}`);
					} catch (error) {
						return fail(String(error));
					}
				}
			}

			// ── settings ─────────────────────────────────────────────────────
			const frames = framesFor(params.duration_seconds ?? 5);
			const settings: Record<string, unknown> = {
				model_type:
					mode === "ref2va"
						? "minimax_h3_ref2va_pruned"
						: turbo
							? "minimax_h3_fl2va_pruned_turbo"
							: "minimax_h3_fl2va_pruned",
				prompt,
				negative_prompt: DEFAULT_NEGATIVE_PROMPT,
				resolution: snapResolution(params.resolution ?? "832x480"),
				video_length: frames,
				num_inference_steps: turbo ? 4 : 20,
				seed: params.seed ?? -1,
				// Fixed by the model, not knobs: H3 has guidance_max_phases=0.
				guidance_scale: 1.0,
				sample_solver: "euler",
				flow_shift: 12.0,
			};
			if (mode === "ref2va") {
				settings.image_refs = refs;
				settings.video_prompt_type = "I";
			} else {
				// Text-only: image_prompt_type must stay unset, not "S"/"E"/"SE".
				settings.image_prompt_type = "";
			}

			// ── submit ───────────────────────────────────────────────────────
			let jobId: string;
			try {
				const response = await fetch(`${SERVER()}/jobs/raw`, {
					method: "POST",
					headers: headers({ "Content-Type": "application/json" }),
					body: JSON.stringify({ settings }),
					signal,
				});
				if (!response.ok) {
					return fail(
						`submit failed: HTTP ${response.status} ${(await response.text()).slice(0, 300)}`,
					);
				}
				const body: any = await response.json();
				jobId = body?.job_id;
				if (!jobId) return fail("submit returned no job_id");
				// Into the transcript immediately: the server evicts finished jobs, so
				// if everything after this fails, this line is the only handle left on
				// a video the GPU may well have produced.
				onUpdate?.({ content: [{ type: "text", text: `submitted job ${jobId}` }] });
			} catch (error) {
				return fail(`could not reach ${SERVER()}: ${error}`);
			}

			// ── poll ─────────────────────────────────────────────────────────
			let files: string[] = [];
			try {
				files = await pollJob(jobId, signal, (text) =>
					onUpdate?.({ content: [{ type: "text", text }] }),
				);
			} catch (error) {
				return fail(String(error), { job_id: jobId });
			}
			if (!files.length) {
				return fail(`job ${jobId} completed but produced no files`, { job_id: jobId });
			}

			// ── download ─────────────────────────────────────────────────────
			try {
				const source = files[0];
				const raw = source.replace(/\/$/, "").split("/").pop() ?? `${jobId}.mp4`;
				// The name comes from the server, and it decides where we write.
				// Keep it to a plain filename so it cannot climb out of the workspace.
				const name = /^[\w.-]+$/.test(raw) && raw !== "." && raw !== ".." ? raw : `${jobId}.mp4`;
				const url = /^https?:\/\//i.test(source) ? source : `${SERVER()}/files/${name}`;

				// The generation is finished and paid for by this point, so a dropped
				// connection here is the most expensive failure in the whole tool.
				let payload: ArrayBuffer | undefined;
				for (let attempt = 1; ; attempt += 1) {
					try {
						const response = await fetch(url, { headers: headers(), signal });
						if (!response.ok) {
							if (retryableStatus(response.status) && attempt < DOWNLOAD_RETRY_LIMIT) {
								throw new TransientHttpError(`HTTP ${response.status}`);
							}
							return fail(`download failed: HTTP ${response.status}`, {
								job_id: jobId,
								url,
							});
						}
						payload = await response.arrayBuffer();
						break;
					} catch (error) {
						if (!isTransient(error, signal) || attempt >= DOWNLOAD_RETRY_LIMIT) {
							return fail(`download of ${url} failed: ${error}`, { job_id: jobId, url });
						}
						onUpdate?.({
							content: [
								{
									type: "text",
									text: `download error ${attempt}/${DOWNLOAD_RETRY_LIMIT}, retrying: ${error}`,
								},
							],
						});
						await sleep(Math.min(POLL_BACKOFF_MS * attempt, POLL_BACKOFF_MAX_MS));
					}
				}

				const destination = join(resolve(WORKSPACE()), name);
				await writeFile(destination, Buffer.from(payload!));
				const bytes = statSync(destination).size;

				return {
					content: [
						{
							type: "text" as const,
							text:
								`Generated ${destination} (${bytes} bytes, ${frames} frames at ${FPS}fps, ` +
								`${settings.resolution}). Audio is already embedded.`,
						},
					],
					details: {
						kind: "generate_video",
						path: destination,
						bytes,
						job_id: jobId,
						mode,
						model_type: settings.model_type,
						frames,
						resolution: settings.resolution,
					},
				};
			} catch (error) {
				return fail(`could not save the generated video: ${error}`);
			}
		},
	});
}

/** Poll until the job finishes, reporting progress. Returns generated file refs. */
async function pollJob(
	jobId: string,
	signal: AbortSignal | undefined,
	report: (text: string) => void,
): Promise<string[]> {
	const deadline = Date.now() + MAX_WAIT_MS;
	let consecutiveFailures = 0;

	for (;;) {
		if (signal?.aborted) throw new Error("cancelled");
		if (Date.now() > deadline) {
			throw new Error(
				`job ${jobId} was still unfinished after ${Math.round(MAX_WAIT_MS / 60000)} minutes`,
			);
		}

		let body: any;
		try {
			const response = await fetch(`${SERVER()}/jobs/${jobId}`, { headers: headers(), signal });
			if (response.status === 404) {
				throw new FatalJobError(`job ${jobId} not found (evicted or invalid)`);
			}
			if (!response.ok) {
				const message = `poll failed: HTTP ${response.status}`;
				if (retryableStatus(response.status)) throw new TransientHttpError(message);
				throw new FatalJobError(message);
			}
			body = await response.json();
			consecutiveFailures = 0;
		} catch (error) {
			if (!isTransient(error, signal)) throw error;
			consecutiveFailures += 1;
			if (consecutiveFailures > POLL_RETRY_LIMIT) {
				// Say the job id out loud: the generation is probably still running
				// server-side, and this is the only handle left for recovering it.
				throw new Error(
					`lost contact with the server after ${POLL_RETRY_LIMIT} consecutive ` +
						`failures (last: ${error}). Job ${jobId} may still be running.`,
				);
			}
			// Node hides the real reason on `.cause`; without it every failure reads
			// as an uninformative "TypeError: fetch failed". Retrying now masks the
			// underlying fault, so the cause has to reach the transcript to stay
			// diagnosable.
			report(
				`poll error ${consecutiveFailures}/${POLL_RETRY_LIMIT}, retrying: ${error}` +
					`${(error as { cause?: unknown })?.cause ? ` (cause: ${(error as { cause?: unknown }).cause})` : ""}`,
			);
			await sleep(Math.min(POLL_BACKOFF_MS * consecutiveFailures, POLL_BACKOFF_MAX_MS));
			continue;
		}

		const status = body?.status;

		if (status === "queued" || status === "running") {
			const pct = Math.round(Number(body?.progress ?? 0) * 100);
			report(
				status === "queued"
					? `queued (position ${body?.queue_position ?? "?"})`
					: `${pct}% ${body?.phase ?? "running"}`,
			);
			await sleep(POLL_INTERVAL_MS);
			continue;
		}

		if (status === "completed" && body?.success) {
			return (body?.generated_files ?? []) as string[];
		}
		throw new Error(
			`job ${status}: success=${body?.success} errors=${JSON.stringify(body?.errors ?? [])}`,
		);
	}
}
