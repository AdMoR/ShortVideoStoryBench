---
name: video-h3-prompting
description: Write the prompt for the generate_video tool, which drives MiniMax H3 — a model that generates video with a synchronized soundtrack in one pass. Use whenever you are about to call generate_video: it covers the production-brief format H3 expects, how to write dialogue, on-screen text and characters, and how to choose duration, resolution and quality.
---

# Prompting MiniMax H3 through `generate_video`

`generate_video` drives MiniMax H3, which produces video **and a synchronized
stereo soundtrack in one pass** — dialogue, ambience and music all come out of
the same call. There is no separate audio step and nothing to remux afterwards.

Everything here is about the `prompt` argument and the four settings around it.
The tool handles submission, waiting and downloading; it blocks until the video
is on disk, and that costs you exactly one turn no matter how long it takes.

## The tool

```
generate_video({
  prompt,            // the production brief — see below. This is the whole game.
  mode,              // "t2va" (text only, default) | "ref2va" (conditioned on images)
  image_refs,        // ref2va only: paths to images in your working directory
  duration_seconds,  // 4-15, default 5. Snapped to the frame grid.
  resolution,        // "WxH", default "832x480". Snapped to multiples of 32.
  quality,           // "turbo" (4-step, default) | "standard" (20-step)
  seed,              // -1 = random (default)
})
```

There is no way to pass an audio or video reference, and no first/last-frame
anchor. Your control over the soundtrack is entirely the prompt text.

## Do not hand-write a terse prompt

H3 expects a structured **production brief** — shots with timestamps, explicit
camera motion, `<d>[Language]…</d>` dialogue tags — not a sentence. A one-line
description under-specifies it and gives visibly worse results.

**[prompt-enrichment-system-prompt.md](prompt-enrichment-system-prompt.md)** is
the format specification: the verbatim system prompt of MiniMax's own
prompt-enrichment step. Read that file and follow its rules as your own
instructions when you write the brief. Do not summarize it from this page — the
field structure is what H3 parses.

The shape it produces:

- **`t2va` (no references)** → a *T2VA brief*: three fields —
  `integrated_multimodal_description`, `overall_soundscape`,
  `non_diegetic_music`.
- **`ref2va` (images attached)** → a *full-reference brief*: six fields —
  `subject_definitions`, `summary`, `retention_analysis`,
  `detailed_description`, `overall_soundscape`, `non_diegetic_music`, labelling
  each image `<Picture N>` / `<Subject N>` **by the order you list it in
  `image_refs`**.

Pass that brief into `prompt` verbatim. It is a structure H3 reads, not prose to
be tightened.

## Name well-known characters; do not rebuild them from description

If a character is recognizable — a franchise character, a public figure — **open
their description with the canonical full name** and make that the identity
anchor. H3 carries a strong learned prior for well-known names. A generic
physical description throws that prior away and forces the model to invent an
anonymous person from scratch on every call; with no memory between calls, that
invention re-rolls independently each time. It is the single biggest cause of a
character's face drifting between shots.

Write `"Michael Scott, the regional manager from The Office (US),"` — not
`"a stocky man in his 40s (Michael), balding, in a rumpled dress shirt"`.

Two rules follow from it:

- **Keep the identity-anchor phrase byte-identical in every shot** the character
  appears in. Even small rewording invites the model to reinterpret the face.
- **Do not invent physical traits on top of the name.** The point of the name is
  to borrow the model's own accurate prior; guessed detail (hair, build, age) is
  redundant at best and fights the name at worst — Michael Scott is not balding.
  Add only detail you actually know, or that the scene requires (a costume, a
  prop).

### Default a named character's dialogue to their original language

A named character speaks their franchise's language unless the brief explicitly
says otherwise. Michael Scott speaks English. A French-market framing is a reason
to make the **on-screen text** French, not a reason to make an American character
speak French. Switch a character's spoken language only when the brief asks for
it directly, never by inferring it from the audience or from surrounding material.

## Dialogue

Dialogue goes in the `detailed_description` field, inline, wrapped in
`<d>[Language]…</d>` at the moment it is spoken. It is generated speech: there is
no reference voice to clone, so the timbre comes from how you describe the
speaker. Describe the voice where you introduce them — "a mature male broadcast
voice, measured pace" — and the model will hold it for the clip.

**Dialogue length must fit the clip.** A line that only just fits gets rushed,
and the first thing to slur is a proper noun. Measured: the longest line in a set
slurred a name at 5.9s and transcribed cleanly at 7.3s. Give a long line more
seconds rather than trimming it — but read the cost table below first, because
those extra seconds are expensive.

**When a line garbles on two seeds, rewrite the line — do not roll a third.**
Both measured failures shared a shape: a short or uncommon word landing in the
rushed tail of the line. Prefer common, concrete vocabulary over terse literary
forms, and move the load-bearing word away from the end. At minutes per take, a
rewrite is also much the cheaper experiment.

Prefer spoken forms that do not collide with their neighbours — "Neymar Jr" is
voiced "Neymar Junior" and ran into the preceding word; plain "Neymar" was clean.

## On-screen text: short strings on a described surface

H3 renders text well when asked for little of it, and degrades in specific,
repeatable ways when asked for more.

- **Keep each string short.** `"#4"` and `"135 M€"` rendered crisply every time,
  € glyph included. The same amount as `"135 MILLIONS D'EUROS"` ran off the frame.
- **One text surface per shot**, described physically — a pitch-side LED board, a
  jumbotron — rather than as a floating graphic. H3 often renders it as a
  full-frame broadcast graphic anyway, which reads better. Don't fight it.
- **Repeated words stutter.** "TOP 5 DES TRANSFERTS" came back as
  "TOP 5 DE DE DE TRANSFERTS" across three `t2va` seeds with the instructions
  tightened each time. The same string rendered perfectly in `ref2va` on a
  jumbotron, first try — **if a title garbles in `t2va`, try `ref2va` before
  re-rolling.**
- **The deterministic fallback is compositing it yourself.** Generate the shot
  with no text at all — "No text, caption, letter or number appears anywhere on
  screen" — and burn the title in with `ffmpeg drawtext`. Size it against the
  frame width: 10 characters of DejaVu Sans Condensed Bold at 86px overflow a
  480px-wide frame; 64px fits.
- **Subjects occlude text.** Say that the subject stands clear of the lettering.
  A dolly-out helps structurally — the text lands fully legible at the wide end.

## References (`mode: "ref2va"`)

`ref2va` conditions the generation on one or more images from your working
directory. Use it to keep a look, a character or a setting across shots.

**A brief may arrive with references already supplied**, sitting in
`references/` in your working directory and listed in the brief with a role and
a description. Those are not decoration: when a brief hands you a face, the
video is scored on whether that face is the one in the shot. Describing the
image in words instead of passing it in `image_refs` is the failure this is
looking for.

**Order is semantic.** `<Picture 1>` is whichever path you list **first** in
`image_refs`. Fix the list and its order before you write the brief, write the
brief against exactly those labels, and do not reorder the list afterwards — a
reordered list silently mislabels every reference in the prompt.

**Never list a reference the shot does not need.** The brief format forbids
asserting media that was given no role, and every unused entry is a chance to
invent a spurious `<Subject N>` that then shows up uncontrolled in
`detailed_description`.

### A style reference should carry no character

When an image is doing *style* work rather than identity work, pick a frame with
**no person in it**. A reference conditions everything it contains, so a
character-bearing frame leaks that face into every shot that shares it.

Worked example: 20 stories about 20 different people shared one `image_refs`
entry — a cel-animated stadium frame with crowd, grass and speed lines but no
face. It held the look across all 120 takes, while each person's identity came
from naming them. Declare the split explicitly, or the model treats the frame as
content:

```text
subject_definitions:
<Picture 1> supplies the reference for the drawing and rendering style of the target
video only … Its stadium, its framing, its camera angle and anything depicted in it
are not part of the target video.

retention_analysis:
<Picture 1> (appears in [Shot 1]): reference - only its hand-drawn cel-animation
rendering style, ink outlines, flat shading and colour treatment are carried over;
its stadium, its framing, its lighting and every object shown in it are replaced.
```

### Use `quality: "standard"` with `ref2va`

The 4-step fast path is a distillation that only the `t2va` model has. On
`ref2va`, `quality: "turbo"` cuts the step count without switching to a
4-step-capable checkpoint, which under-denoises the result. Pass
`quality: "standard"` for any `ref2va` call you intend to keep.

## Duration is the expensive setting, and it is non-linear

`duration_seconds` is snapped onto a frame grid — 24fps, 107 frames minimum, in
steps of 17 — so the real choices are 4.5s, 5.2s, 5.9s, 6.6s, 7.3s and up. The
tool tells you what it snapped to.

Measured back to back, same brief, same server:

| frames | duration | wall clock |
|---|---|---|
| 124 | 5.2s | ~14 min |
| 175 | 7.3s | ~85 min |

That is roughly **6x the cost for 1.4x the duration** — the shape of a memory
cliff, not of linear scaling. So when the brief needs N seconds of video, **buy
them with more shots, not longer ones.** Keep takes at or below ~5.2s unless a
single unbroken shot is genuinely required, and time one before committing to a
batch of longer ones.

This is the setting most likely to cost you the whole run. A story assembled from
five 5-second takes is affordable; the same story as three 7.5-second takes is
not.

## Resolution: multiples of 32, and what the extra pixels buy

Both dimensions must be multiples of 32, which rules out the nominal vertical
format — **720x1280 is invalid**, 720 is not a multiple of 32. The nearest valid
9:16 sizes are `704x1248`, `736x1312`, `608x1088`. The tool snaps for you, but
pick deliberately: it snaps 720 down to 704 and leaves you off-ratio if you also
passed 1280.

The default `832x480` is landscape. For a vertical short, pass `480x832`.

Measured at 30 steps:

| resolution | ~time per 6s take | notes |
|---|---|---|
| 480x832 | ~15 min | the model's comfortable size |
| 704x1248 | ~40 min | 2.2x the pixels, noticeably better faces |

The gain is mostly **identity fidelity**: at 480x832 a well-known face drifted
into a look-alike after a cut; at 704x1248 the same prompt kept the likeness.
Text rendering was already good at 480x832 and did not improve. Budget the time
before committing a batch, and look at the first take rather than discovering the
problem five takes in.

## Camera direction is worth choosing, not defaulting

Opening tight and dollying **out** reveals the scene and any on-screen graphic as
the frame opens, which lands the reveal on the move. Opening wide and pushing in
shows the graphic from frame 0 and wastes the beat. Write the move into the brief
either way — "explicit camera motion" is one of the things the brief format asks
for, and leaving it out gets you a static shot.

## Check the video you got, before you build on it

The soundtrack is the one output you cannot check by looking at frames, and it
fails silently — a take with perfect visuals can be speaking the wrong words. If
you have a way to transcribe it, do; if you do not, at least confirm the file is
a real video (`ffprobe`) and long enough before assembling anything on top of it.

Judge dialogue on **content**, not on exact wording: a transcriber writes
"Numéro 4 … 135 millions" where the script says "Numéro quatre … cent
trente-cinq millions". Proper nouns transcribe badly even when spoken correctly —
treat a name-only mismatch as inconclusive rather than as a defect, and do not
spend a re-roll on it.
