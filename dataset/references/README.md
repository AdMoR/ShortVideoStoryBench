# Reference images

One directory per seed; `dataset/seeds.yaml` points at the files by a path
relative to `dataset/`. They are staged into the agent's workspace as
`references/<reference id><suffix>` and shown to the judge ahead of the clip's
frames, so what is in them is measured by `S5 — Reference Adherence`.

**The images committed here are flat placeholder illustrations, not photographs.**
They exist so the whole path — staging, brief, `ref2va`, judging — runs and can
be tested offline without shipping a licensed photo in the repo. They are good
enough to check that an agent *used* a reference and that identity drift is
visible; they are not good enough to measure how well a model holds a real face.
Replace them with real images before quoting an S5 number.

Keep them around 768px on the long edge: the judge downscales to 1024
(`judge/frames.py:load_image`) and the video server re-encodes anyway, so
anything larger is repo weight for nothing.
