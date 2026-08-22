/**
 * video-eval-bench handoff tool.
 *
 * The agent produces a video somewhere in its workspace and calls `submit_video`
 * with the path. This tool copies it to the destination the benchmark pinned into
 * the environment at spawn time, and terminates the turn.
 *
 * Two things are deliberate:
 *
 *   * The destination is VEB_OUTPUT_PATH from the environment, never a tool
 *     parameter. The model has no way to express "write it somewhere else", so no
 *     prompt can redirect the output. It also means the agent never has to remember
 *     the destination — which is what keeps the handoff working after pi's
 *     auto-compaction has summarized the original brief away.
 *
 *   * A rejection returns isError WITHOUT `terminate`, so the agent stays in the
 *     loop and can fix the problem. Only a successful submission ends the turn.
 *
 * No @earendil-works/pi-tui import: the benchmark runs pi headless in --mode json,
 * where renderResult never fires and terminal components are dead weight.
 */

import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { dirname, extname, isAbsolute, resolve, sep } from "node:path";
import { Type } from "typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const OUTPUT_PATH = () => process.env.VEB_OUTPUT_PATH ?? "";
const WORKSPACE = () => process.env.VEB_WORKSPACE ?? process.cwd();

// Container formats the judge's frame extractor (OpenCV) can open.
const VIDEO_EXTENSIONS = new Set([".mp4", ".webm", ".mov", ".mkv", ".avi"]);

/** Resolve a user-supplied path against the workspace, refusing to escape it. */
function resolveInWorkspace(path: string): { ok: true; path: string } | { ok: false; why: string } {
	const workspace = resolve(WORKSPACE());
	const candidate = isAbsolute(path) ? resolve(path) : resolve(workspace, path);
	if (candidate !== workspace && !candidate.startsWith(workspace + sep)) {
		return { ok: false, why: `path is outside the working directory (${workspace})` };
	}
	return { ok: true, path: candidate };
}

function reject(why: string) {
	return {
		content: [
			{
				type: "text" as const,
				text: `submit_video rejected: ${why}. Fix the problem and call submit_video again.`,
			},
		],
		details: { kind: "submit_video_error", error: why },
		isError: true,
		// No `terminate` — the agent must stay in the loop to retry.
	};
}

export default function benchTools(pi: ExtensionAPI) {
	pi.registerTool({
		name: "submit_video",
		label: "Submit Video",
		description:
			"Deliver the finished video for this brief. Pass the path of the video file you " +
			"produced. This ends the task, so call it only once, on your final video.",
		promptSnippet: "Deliver the finished video file as the final result",
		promptGuidelines: [
			"Call submit_video with the path of your finished video as your final action — a file left in the working directory is not a submission.",
			"If submit_video returns an error, fix the reported problem and call submit_video again.",
		],
		parameters: Type.Object({
			path: Type.String({ description: "Path to the finished video file" }),
			notes: Type.Optional(
				Type.String({ description: "Optional one-line note about what was produced" }),
			),
		}),

		async execute(_toolCallId, params) {
			const destination = OUTPUT_PATH();
			if (!destination) {
				return reject("VEB_OUTPUT_PATH is not set — the benchmark harness is misconfigured");
			}

			const resolved = resolveInWorkspace(params.path);
			if (!resolved.ok) return reject(resolved.why);
			const source = resolved.path;

			let size: number;
			try {
				const stat = statSync(source);
				if (!stat.isFile()) return reject(`${source} is not a file`);
				size = stat.size;
			} catch {
				return reject(`${source} does not exist`);
			}
			if (size === 0) return reject(`${source} is empty (0 bytes)`);

			const extension = extname(source).toLowerCase();
			if (!VIDEO_EXTENSIONS.has(extension)) {
				return reject(
					`${source} is not a video file (extension ${extension || "none"}; ` +
						`expected one of ${[...VIDEO_EXTENSIONS].join(", ")})`,
				);
			}

			try {
				mkdirSync(dirname(destination), { recursive: true });
				copyFileSync(source, destination);
			} catch (error) {
				return reject(`could not copy ${source} to the output location: ${error}`);
			}

			// The copy is complete before this returns, and tool_execution_end is
			// emitted after execute() resolves — so by the time the harness sees the
			// event, the file is whole. That is what makes its early exit safe.
			return {
				content: [
					{
						type: "text" as const,
						text: `Submitted ${source} (${size} bytes). The task is complete.`,
					},
				],
				details: {
					kind: "submit_video",
					path: destination,
					source,
					bytes: size,
					notes: params.notes ?? "",
				},
				terminate: true,
			};
		},
	});
}
