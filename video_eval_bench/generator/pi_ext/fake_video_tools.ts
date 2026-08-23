/**
 * Stand-in for the WanGP `generate_video` tool.
 *
 * Same tool name, same parameters, same result shape as wangp_tools.ts — the
 * agent cannot tell the difference. It copies a fixed video into the working
 * directory and returns immediately.
 *
 * This exists so the expensive question and the cheap one can be asked
 * separately. Whether the agent *drives the tool correctly* — right mode, a
 * full brief in the prompt, submitting the result afterwards — is answerable in
 * seconds and does not need a GPU. Only whether the video is any *good* needs
 * the real backend. Running the cheap question against the real one wastes
 * minutes per seed and makes prompt iteration impractical.
 *
 * Knobs (env):
 *   VEB_FAKE_VIDEO       path to the video handed back (defaults to the bundled asset)
 *   VEB_FAKE_VIDEO_DELAY seconds to stall, to exercise the harness's long-wait paths
 *   VEB_FAKE_VIDEO_FAIL  when "1", fail the first call, to see whether the agent retries
 */

import { copyFileSync, existsSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Type } from "typebox";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const WORKSPACE = () => process.env.VEB_WORKSPACE ?? process.cwd();
const DELAY_MS = () => Number(process.env.VEB_FAKE_VIDEO_DELAY ?? "0") * 1000;
const FAIL_FIRST = () => process.env.VEB_FAKE_VIDEO_FAIL === "1";

function bundledAsset(): string {
	const here = dirname(fileURLToPath(import.meta.url));
	return join(here, "assets", "fake_take.mp4");
}

function sourceVideo(): string {
	return process.env.VEB_FAKE_VIDEO || bundledAsset();
}

let calls = 0;

export default function fakeVideoTools(pi: ExtensionAPI) {
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
			calls += 1;
			const prompt = (params.prompt ?? "").trim();
			if (!prompt) {
				return {
					content: [{ type: "text" as const, text: "generate_video failed: prompt is empty" }],
					details: { kind: "generate_video_error", error: "prompt is empty" },
					isError: true,
				};
			}
			if ((params.mode ?? "t2va") === "ref2va" && !params.image_refs?.length) {
				const why = "mode 'ref2va' needs at least one path in image_refs";
				return {
					content: [{ type: "text" as const, text: `generate_video failed: ${why}` }],
					details: { kind: "generate_video_error", error: why },
					isError: true,
				};
			}
			if (FAIL_FIRST() && calls === 1) {
				const why = "the generation server returned HTTP 500 (injected failure)";
				return {
					content: [{ type: "text" as const, text: `generate_video failed: ${why}` }],
					details: { kind: "generate_video_error", error: why },
					isError: true,
				};
			}

			const delay = DELAY_MS();
			if (delay > 0) {
				onUpdate?.({ content: [{ type: "text", text: "generating…" }] });
				await new Promise((r) => setTimeout(r, delay));
			}
			if (signal?.aborted) {
				return {
					content: [{ type: "text" as const, text: "generate_video cancelled" }],
					details: { kind: "generate_video_error", error: "cancelled" },
					isError: true,
				};
			}

			const source = sourceVideo();
			if (!existsSync(source)) {
				const why = `the stand-in video ${source} is missing`;
				return {
					content: [{ type: "text" as const, text: `generate_video failed: ${why}` }],
					details: { kind: "generate_video_error", error: why },
					isError: true,
				};
			}

			const destination = join(resolve(WORKSPACE()), `take_${calls}.mp4`);
			copyFileSync(source, destination);
			const bytes = statSync(destination).size;

			return {
				content: [
					{
						type: "text" as const,
						text:
							`Generated ${destination} (${bytes} bytes, ${params.duration_seconds ?? 5}s, ` +
							`${params.resolution ?? "832x480"}). Audio is already embedded.`,
					},
				],
				details: {
					kind: "generate_video",
					path: destination,
					bytes,
					job_id: `fake-${calls}`,
					mode: params.mode ?? "t2va",
					prompt_chars: prompt.length,
					stand_in: true,
				},
			};
		},
	});
}
