from pathlib import Path
from typing import Awaitable, Callable, Protocol, runtime_checkable

from video_eval_bench.dataset.seed import Seed

# A generator: given a seed and the run's output directory, produce a video
# and return its local path. `MockGenerator` (mock_generator.py) is the
# offline implementation used for tests and skeleton validation.
GenerateFn = Callable[[Seed, Path], Awaitable[str]]


@runtime_checkable
class GeneratorWithMetadata(Protocol):
    """
    Optional: a generator that can say how it produced each video.

    `run_bench` checks for this and files whatever comes back on the seed's
    result, so an agentic generator can surface its transcript path, turn count
    and token usage in the report without the bench loop knowing what any of
    those mean.
    """

    def metadata_for(self, seed_id: str) -> dict: ...
