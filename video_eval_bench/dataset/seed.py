from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field


# What a reference image is *for*. An enum rather than free text because the role
# is the one thing consumers branch on: the h3 skill treats an identity reference
# and a style reference as different objects (a style reference must carry no
# character), and the judge asks a different question of each.
ReferenceRole = Literal["character", "location", "style", "prop"]


class SeedReference(BaseModel, extra="forbid"):
    """
    One image supplied with a brief: a face to hold to, a place, a look, an object.

    `path` is written relative to the dataset directory and resolved to an
    absolute path by `load_seeds`, so everything downstream — the generator
    staging it into a workspace, the judge loading it as JPEG bytes — gets a path
    it can open without carrying the dataset directory around.

    `extra="forbid"` is deliberate: `Seed` itself has no extra setting, so a
    misspelled key there is silently dropped. For a field that decides whether an
    image reaches the agent at all, silence is the wrong failure.
    """

    id: str = Field(
        # Constrained because the id is joined onto a path: it is the filename
        # the image is staged under in the agent's workspace, so a separator or a
        # `..` in it would write outside the directory it was meant to land in.
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        description="Stable slug, unique within the seed. Becomes the staged filename.",
    )
    role: ReferenceRole = Field(description="What the image is for")
    label: str = Field(description="Human name to use in prompts, e.g. 'Maya'")
    description: str = Field(description="What the generated video must hold to")
    path: Path = Field(description="Image path, relative to the dataset directory")


class Seed(BaseModel):
    """One benchmark case: generate a video from this prompt, then judge it."""

    seed_id: str = Field(description="Unique id, e.g. 'entertainment_001'")
    category: str = Field(description="Genre key — selects the Section C rubric")
    prompt: str = Field(description="Text prompt used to generate the video")
    references: List[SeedReference] = Field(
        default_factory=list,
        description=(
            "Reference images supplied with the brief. Order is meaningful: the "
            "agent maps them onto the video model's reference slots in the order "
            "it lists them."
        ),
    )
    metadata: dict = Field(default_factory=dict)
