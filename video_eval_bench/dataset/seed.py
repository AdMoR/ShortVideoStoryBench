from typing import Awaitable, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel, Field



class Seed(BaseModel):
    """One benchmark case: generate a video from this prompt, then judge it."""

    seed_id: str = Field(description="Unique id, e.g. 'entertainment_001'")
    category: str = Field(description="Genre key — selects the Section C rubric")
    prompt: str = Field(description="Text prompt used to generate the video")
    reference_video: Optional[str] = Field(
        default=None,
        description="Optional path/URL to a reference video the generation should match",
    )
    metadata: dict = Field(default_factory=dict)


