# Writing and reviewing criteria

The standard `rubrics.yaml` is held to. Read it before adding a criterion, before
merging two, and before deciding a criterion that "feels vague" is fine because it
has always been there.

Everything here came out of a line-by-line review of the library. Most of the
criteria in the file today are second drafts, and the rules below are what the first
drafts got wrong. The library is a small file that decides what a benchmark measures,
so a criterion that cannot be graded does not merely add noise — it hands out weight
for free, and it does it to whichever seeds it fits worst.

## The rules

### 1. A criterion is a question two judges answer the same way

Write what to look at and what fails. If the check turns on an adjective — good,
appealing, clean, well-composed, professional, engaging, appropriate — either give it
a threshold or do not write it.

> "Each element stays on screen long enough to be read." — long enough is *what*?
> Rewritten as PACE1: about a second for a short line of text, never under half a
> second, and nothing replaced before it resolves.

> "The palette delivers the mood the brief named." — became MOOD1: applies only when
> the brief names the look in so many words, binds that phrase, and asks whether the
> dominant cast and key light deliver it over most of the runtime.

The test is not whether *you* could grade it. It is whether two judges shown the same
clip, on two different days, land in the same place.

### 2. Say when it applies

A criterion that does not state its gate will be answered anyway — "PASSED, not
applicable" — and bank its full weight. That is the single failure this dataset was
restructured around.

So every criterion whose question only exists for a kind of brief opens by saying so:

- SETUP1 / TENS1 / CAUSE1 / REVEAL1 — only for a brief that promises a change and the
  work that causes it. They are unanswerable for a montage, a talking head, an
  explainer, which is why they now sit in their own `transformation` section.
- HIST1 — only when the brief names a period. "Realistic" and "modern day" are not
  periods; a present-day clip has nothing to be anachronistic against.
- CULT1 — only when the brief names a symbol whose depiction can be checked. Every
  video is set in some culture; that is not what it asks.
- DIAG1 — only when the brief calls for a diagram on screen.
- HIGHL1 — only for a brief built around one named moment.
- FACT1 — only with a stated, checkable claim bound to it.

A gate is not a hedge. It is the difference between a criterion and a mood.

### 3. Bind what varies

When the thing being graded is "whatever the brief said", make the seed say it. A
`{placeholder}` in the description, declared in `binds`, filled in by the seed:

```yaml
- id: MOOD1
  bind: {look: "a warm golden grade, held through all four shots"}
```

This is what turned FACT1 from "the video should be true" into "does the clip depict
*this claim* correctly". Same id, same weight, same report column, a question the
judge can actually answer. The criteria that bind today are, without exception, the
ones a reviewer called vague: MOOD1 (`look`), FACT1 (`claim`), TRIG1 (`trigger`),
SETUP1 (`before`), TENS1 (`stakes`), REVEAL1 (`payoff`), HIST1 (`period`), CULT1
(`symbol`). That is not a coincidence — "it should match the brief" becomes gradeable
exactly when the seed has to write down what the brief said.

### 4. One question per criterion — and merge the ones that are the same question

Two criteria a judge would answer from the same look at the same frame are one
criterion with two names. They double the weight of one failure and disagree with
each other in the report.

Merged this way, and the reasoning each time:

| Was | Now | Why |
| --- | --- | --- |
| `PROP1` (named props) + `CTX1` (background context) + `SEC1` | `SEC1` Secondary Element Consistency | All three ask what happens to what is *not* the main subject. One failure mode — the generator does not track what it is not asked about — asked three times with narrower scopes. |
| `CLEAN1` (no distractors) + `FOCUS1` (focus clarity) | `FOCUS1` Subject Emphasis | "Nothing irrelevant is on screen" and "the subject is what the frame emphasises" cannot be graded apart. A busy frame is not a failure; a busy frame that buries the subject is. |
| `STEP1` (step order) | `SEQ1` Scene & Step Sequence | Four named steps is an enumeration like three numbered scenes. SEQ1 was already asking it. |
| `EFFECT1` in `interactive` | `EFFECT1` in `general` | It started as a games criterion and is not one: any brief where something is done *to* something can fail it. |
| `ANIM1` in `interactive` | `ANIM1` in `general`, on every seed | "The clip actually moves" is not a property of animated briefs. Every brief asks for a video, so every seed can fail it — the only criterion in the library that is on all of them. |
| `PHYS1` in `narrative` | `PHYS1` in `general` | Any clip can break the physics of its own scene, whether or not it tells a story. |

The opposite move is just as important: two criteria that *look* alike stay apart when
they catch different failures, and each says so in its own text. ART1 vs TRANS1 (smear
inside a shot vs. blend between shots), UI1 vs UIVAL1 (a HUD drawn vs. a HUD telling
the truth), ANIM1 vs NATURAL1 (is anything moving vs. does the actor move like a
person), MOMENT1 vs HIGHL1 (is the moment present vs. is the clip built around it),
CAUSE1 vs EFFECT1 (the arc's middle vs. a single action's consequence). If you cannot
write that sentence for a new criterion, you are adding a duplicate.

### 5. Delete what cannot be rescued

Not every criterion can be gated or bound into shape. `GEO1` asked for "the right
shapes in the right positions" of named landmarks and gave no judge a way to decide
what right was. It was removed, and the one brief that genuinely turned on landmarks
got a **local criterion** instead — `marketing_002.LANDMARK1`, which can afford to name
the four landmarks and the one feature each that identifies it.

That is the escape hatch for anything too specific to generalise: a `local_criteria`
entry on the seed, namespaced to it. A criterion belongs in the library only if many
different briefs, about different subjects, in different genres, could carry it.

### 6. Span: how much of the clip does it need?

The `span` tags — `single_shot`, `cross_shot`, `whole_clip` — are what separate the
general criteria from the ones that only apply to a kind of video, and a criterion can
carry more than one:

- **SUBJ1** is `cross_shot` *and* `whole_clip`. A subject drifts across a cut, and it
  drifts inside an unbroken take with no cut to hide behind. Tagging it cross-shot
  alone implied a single-take brief could not fail it, which is backwards — that is
  the harder case.
- **SEC1** is both for a different reason: its "is the scene dressed at all" half is
  answerable from any clip, while its "does it stay the same" half needs the element
  on screen twice.
- Everything else in `continuity` needs a second shot of the same material and says so.

Where a criterion sits and what it is tagged decide nothing at judge time. They decide
what a reviewer sees when reading the file, and what a report can group by — which is
exactly why they should be right.

Tag one criterion with every subject it can legitimately be about, not the one you had
in mind when writing it. SEC1 carries `people`, `object` *and* `environment`: a
background actor, a passing vehicle and the wallpaper are all things it catches, and a
report filtered to `people` that hid it would be lying about the library's coverage.

### 7. Broad subjects, and the niches inside them

The `subject` group holds the broad themes a check can be about — `people`,
`environment`, `object`, `text`, `audio`, `camera`, `colour`, `lighting`, `story`,
`visual_effects`. `sub_theme` holds the narrow ones that live inside those: `brand`
(a marketing-only flavour of object and text), `diagram` (a figure on screen) and
`interface` (a HUD or a dashboard, which only game and screen-capture briefs have).
Two rules follow:

- A sub-theme never stands alone. Every criterion carrying one also carries the broad
  subject it sits inside, so filtering the library by `text` still finds `diagram`
  criteria and filtering by `object` still finds `brand` ones.
- A report lists the two groups separately. Mixed together, `brand`, `diagram` and
  `interface` read as peers of `people` and `environment`, which makes a library with
  five marketing criteria look like it is a fifth about branding, and two HUD checks
  look like an interface benchmark.

The test for which group a tag belongs in is how many briefs could carry it. `people`
is a theme any brief might touch; `interface` is one a game or screen-capture brief
has and nothing else does. A tag that is really a genre in disguise is a sub-theme.

Name a tag for what it is, not for what it is adjacent to: `visual_effects` rather
than `effects` (which read as "anything effectful"), and one `audio` rather than
`music` plus `speech` — splitting one axis in half means no single number answers
"how much of this benchmark is about sound", and it leaves the criteria that are about
neither, like the ambience bed under both, with nowhere to sit. `anatomy` went the same
way, into `people`: it was a property of one criterion, not a theme of the library.

### 8. Sections: general first, narrow ones for gated criteria

A section says what *kind* of check lives there. It is not a bundle, it attaches
nothing to any seed, and a criterion belongs to exactly one.

The order is generic → specific, and "specific" now means something concrete: a
section whose criteria all carry a gate. `transformation` (a brief with a before and
an after), `references` (a brief that ships reference images — the one section with a
hard activation rule, `requires_references: true` on every criterion in it),
`attention` (a brief that says what a shot or a clip is about), `world` (a brief that
names a period or a symbol). If a criterion needs a gate, it probably wants a narrow
section, and if a narrow section has one member that is fine.

`light` exists for the opposite reason: exposure and grade were filed under
`instructional`, next to text legibility, where nobody would look for them.

Sections merge for the same reason criteria do. `music` was folded into `audio`: two
sections of three and five, split on a distinction that changed nothing about how the
checks are written or judged, and which left ACONT1 — the ambience under speech *and*
music — belonging to neither.

## Adding a criterion: the checklist

1. Could many different briefs, in different genres, about different subjects, carry
   it? If not, make it a `local_criteria` entry on the seed.
2. Is it one observable fact? If stating it needs "and", it is two.
3. Would two judges agree, every time? Name the thresholds.
4. When does it apply, and does the text say so?
5. Does anything vary per seed? Bind it.
6. Which existing criterion is it closest to, and can you write the sentence that
   separates them? Put that sentence in the description.
7. `evidence`: what must a validator *see* — `description`, `pixels`, `motion`,
   `audio`, `container`? This routes the seed builder's verification; a `pixels`
   question asked of a text judge gets a confident answer that means nothing.
8. Tags from the closed vocabulary, including the right `span`.
9. Section: where would a reviewer look for it first?

A criterion reaches no seed by being added. The seed builder's `select` stage picks it
per brief, grounded in that brief, and both of its judges check the choice.
