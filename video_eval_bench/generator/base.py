"""
The generator contract.

A generator answers one question per seed: *is there a video for this, and what
do you know about it?* It returns a `GenerationResult` rather than a bare path
because the answer is rarely just a filename — an agentic run knows how many
turns and tokens it burned, an imported video knows which tool made it and what
that cost, and an import may have no video for a seed at all.

    async def generate(seed: Seed, output_dir: Path) -> GenerationResult

`MockGenerator` (mock_generator.py) is the offline implementation used for tests
and skeleton validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Literal, Optional

from video_eval_bench.dataset.seed import Seed

# What a generator may report. Failure is not in here: a generator signals it by
# raising GenerationError, and `run_bench` records that as status "errored". So a
# *result* has three states and a generator can only produce two of them.
GenerationStatus = Literal["completed", "skipped"]


@dataclass
class GenerationResult:
    """
    What a generator produced for one seed
    """

    seed_id: str
    status: GenerationStatus = "completed"
    video_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    duration_seconds: Optional[float] = None


class GenerationError(RuntimeError):
    """
    Generation failed for one seed. run_bench records it and continues.
    """

    def __init__(self, message: str, metadata: Optional[dict] = None):
        super().__init__(message)
        self.metadata = metadata or {}


GenerateFn = Callable[[Seed, Path], Awaitable[GenerationResult]]
