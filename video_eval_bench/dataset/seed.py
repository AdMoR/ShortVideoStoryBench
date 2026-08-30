from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# What a reference image is *for*. An enum rather than free text because the role
# is the one thing consumers branch on: the h3 skill treats an identity reference
# and a style reference as different objects (a style reference must carry no
# character), and the judge asks a different question of each.
ReferenceRole = Literal["character", "location", "style", "prop"]


# The two judges of the seed builder, as they land on a criterion.
#
# `SeedJudgeVerdict` answers "could a video generated from this seed alone satisfy
# this criterion, and could a competent attempt still fail it". `CriterionVerification`
# answers "does the source material this seed was derived from satisfy it".
#
# Neither ever removes a criterion. They annotate; `scored` — set by the builder's
# policy at emit time — is the only thing that decides what the benchmark grades on.
# A criterion rejected and deleted teaches nothing, and the whole point of the build
# report is to read the rejections grouped by cause.
SeedJudgeStatus = Literal["grounded", "ungrounded", "unfailable"]
VerificationStatus = Literal["verified", "contradicted", "undetermined", "unchecked"]
VerifiedBy = Literal["metadata", "container", "video", "none"]


class SeedJudgeVerdict(BaseModel, extra="forbid"):
    """Whether the seed's own brief gives this criterion something to bite on."""

    status: SeedJudgeStatus
    reason: str = ""


class CriterionVerification(BaseModel, extra="forbid"):
    """
    Whether the material the seed was derived from satisfies this criterion.

    `by` names the validator that reached the verdict, and `status="unchecked"` with
    `by="none"` is a first-class outcome rather than a failure: a criterion whose
    evidence class no validator in the run could see was never asked, and recording
    that as a failure would read as a defect in a criterion nobody tested.
    """

    status: VerificationStatus
    by: VerifiedBy = "none"
    comment: str = ""
    reason: str = ""


class SeedCriterion(BaseModel, extra="forbid"):
    """
    One criterion a seed carries, with the values that bind it to this seed.

    `bind` supplies the placeholders the library criterion declares in its `binds`
    list, so a single generic criterion can ask about *this* seed's subject without
    forking into a per-seed criterion — the id, the weight and the dimension stay
    shared, and scores stay comparable across every seed that carries the id.

    `extra="forbid"` for the same reason `SeedReference` has it: a misspelled key
    here decides whether a criterion is graded at all, and silence is the wrong
    failure for that.
    """

    id: str = Field(description="Criterion id from the rubric library")
    bind: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Values for the placeholders the criterion declares in `binds`. "
            "Validated against them at load: a key the criterion does not declare, "
            "or a declared placeholder left unbound, fails the whole load."
        ),
    )
    scored: bool = Field(
        default=True,
        description=(
            "Whether the judge grades this criterion. Set by the seed builder's "
            "policy at emit time; hand-written seeds leave it True. A criterion "
            "excluded by policy stays in the file with its verdicts attached — the "
            "dataset is lossless and the policy can be changed without a rebuild."
        ),
    )
    seed_judge: Optional[SeedJudgeVerdict] = None
    verification: Optional[CriterionVerification] = None


class SeedProvenance(BaseModel, extra="forbid"):
    """Where a generated seed came from, and what produced it."""

    source: str = Field(description="Source dataset, e.g. 'finevideo'")
    sample_id: str = Field(description="Id within the source, e.g. 'sample_1000'")
    youtube_id: str = ""
    source_category: str = Field(default="", description="Source's own category, verbatim")
    duration_seconds: int = 0
    digest_sha256: str = Field(default="", description="Hash of the digest the build read")
    builder_version: str = ""
    prompt_hashes: Dict[str, str] = Field(
        default_factory=dict,
        description="sha256 of each builder prompt, so a seed is attributable to a revision",
    )


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
    category: str = Field(
        description=(
            "Genre key — a reporting label only. It groups the seed in summaries "
            "and names the kind of brief to the judge; `rubrics` selects what the "
            "seed is actually judged on."
        )
    )
    rubrics: List[SeedCriterion] = Field(
        min_length=1,
        description=(
            "The criteria from the rubric library (dataset/rubrics.yaml) this seed "
            "carries, and only those. Required, with no default: a seed whose list "
            "was forgotten must fail the load, because an empty rubric would "
            "otherwise score the video 0/0 and report it as a clean pass. A bare "
            "id string is accepted as shorthand for `{id: <it>}`."
        ),
    )
    prompt: str = Field(description="Text prompt used to generate the video")
    references: List[SeedReference] = Field(
        default_factory=list,
        description=(
            "Reference images supplied with the brief. Order is meaningful: the "
            "agent maps them onto the video model's reference slots in the order "
            "it lists them."
        ),
    )
    local_criteria: List["RubricCriterion"] = Field(
        default_factory=list,
        description=(
            "Criteria defined on this seed alone — tier 3 of the rubric hierarchy. "
            "For a check this one brief puts at risk that no library section covers "
            "and no other brief needs, so promoting it to the library would be "
            "generalising from one example. Ids must be namespaced to the seed "
            "(`fv_6146.LANG1`), and each must also appear in `rubrics` to be judged."
        ),
    )
    tags: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Facet -> values, from the dataset's tags.yaml vocabulary. A closed "
            "vocabulary rather than free text because these are the columns every "
            "cross-seed analysis groups by, and drifting values would silently "
            "split one column into three."
        ),
    )
    provenance: Optional[SeedProvenance] = Field(
        default=None, description="Set on generated seeds; None on hand-written ones"
    )
    metadata: dict = Field(default_factory=dict)

    @field_validator("rubrics", mode="before")
    @classmethod
    def _accept_bare_ids(cls, value):
        """
        Let a seed write `- SUBJ1` where it has nothing to bind.

        This is a shorthand at the parse boundary, not a second representation:
        everything downstream of the model sees `SeedCriterion` and only that. The
        hand-written seeds carry no bindings and no verdicts, and spelling them as
        fourteen-line mappings would bury the one thing those lists are for.
        """
        if not isinstance(value, list):
            return value
        return [{"id": item} if isinstance(item, str) else item for item in value]

    @field_validator("tags", mode="before")
    @classmethod
    def _accept_scalar_tag_values(cls, value):
        """`pacing: fast` means `pacing: [fast]` — single-valued facets are the norm."""
        if not isinstance(value, dict):
            return value
        return {k: [v] if isinstance(v, str) else v for k, v in value.items()}

    def criterion_ids(self) -> List[str]:
        """Every criterion id the seed carries, scored or not."""
        return [c.id for c in self.rubrics]

    def scored_criteria(self) -> List[SeedCriterion]:
        """The criteria the judge should actually grade."""
        return [c for c in self.rubrics if c.scored]

    def local_criteria_by_id(self) -> Dict[str, "RubricCriterion"]:
        """This seed's own criterion definitions, by id."""
        return {c.id: c for c in self.local_criteria}


# Resolved here rather than by importing at the top of the module: `dataset_schemas`
# imports `Seed`, so the arrow only points one way at import time. `RubricCriterion`
# is a plain pydantic model with no dependency back on this file, which is what makes
# the late rebuild safe.
from video_eval_bench.dataset.dataset_schemas import RubricCriterion  # noqa: E402

Seed.model_rebuild()
