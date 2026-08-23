# Brief

{prompt}

## Details

- Genre: {category}
- Working directory: {workspace}

Produce one video that satisfies the brief above, then call `submit_video` with its
path. Work inside the working directory.

## Time

You have about {budget_minutes} minutes of wall-clock time for this task, and no
clock of your own — generation and encoding both take real minutes, so spend the
budget deliberately.

**Submit the first usable video you have, before you improve anything.** As soon
as any video file exists that is even roughly on-brief — a single raw generated
clip counts — call `submit_video` on it. Then keep working and call
`submit_video` again each time you have something better.

The last submission is the one that counts, so an early one costs you nothing and
can only protect you. Do not wait until the edit is finished: the run can end
sooner than you expect, and a perfect video still sitting in the working directory
scores nothing at all. Treat every submission as a checkpoint, not as a finish
line.
