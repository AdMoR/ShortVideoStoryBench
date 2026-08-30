"""
Dataset loading: seeds + the rubric library, from YAML files.

Layout:
    dataset/
        rubrics.yaml    — the criterion library, organised into scoped sections
        genres.yaml     — the genre vocabulary
        safety.yaml     — binary veto checks, applied to every video
        seeds.yaml      — the seeds, each naming the criteria that apply to it

The library is organised into sections, generic first, and every criterion carries
tags from a closed vocabulary. Both are analytic axes: they group criteria for
reading and reporting and select nothing. A seed may also define `local_criteria`
of its own, for a check no other brief needs.

A seed's rubric is the list it names and nothing more. `Dataset.criteria_for`
reads the seed, never the sections — no category is applied on top, because a
criterion the brief cannot fail would then be answered "PASSED, not applicable"
and still count its full weight.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from video_eval_bench.dataset.seed import Seed

logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"


class Dimension(BaseModel):
    """A reporting group for criteria. Grouping only — scoring is flat."""

    key: str = Field(description="Dimension key, e.g. 'consistency'")
    name: str = Field(description="Human-readable name")
    description: str = Field(default="", description="What the dimension covers")


# What a validator must be able to see to adjudicate a criterion.
#
# This is what routes verification in the seed builder: a judge reading a *description*
# of a video can settle "do the shots progress in the order the brief asked for", and
# cannot settle "are there generation artifacts" — the description will never mention
# artifacts, so it would answer confidently and wrongly, and a perfectly good criterion
# would come back looking contradicted by its own source material.
#
#   description — a semantic/narrative fact any faithful description captures
#   pixels      — needs to look at frames
#   motion      — needs the clip moving, not stills
#   audio       — needs the audio track
#   container   — a file-level property, decidable exactly by ffprobe with no model
Evidence = Literal["description", "pixels", "motion", "audio", "container"]


class RubricCriterion(BaseModel):
    """One observable (binary) check a seed can be judged on."""

    id: str = Field(description="Stable id, unique across the library, e.g. 'SUBJ1'")
    dimension: str = Field(description="Dimension key — groups the criterion in reports")
    name: str = Field(description="Short criterion name")
    description: str = Field(description="What to check, in detail")
    weight: float = Field(gt=0.0, description="Points awarded when the check passes")
    critical: bool = Field(
        default=False,
        description="True for ⚠️ criteria — the ones a video must not fail",
    )
    evidence: Evidence = Field(
        description=(
            "What a validator must be able to see to settle this criterion. "
            "Required, with no default: it decides which validator the seed builder "
            "may ask, and a wrong guess sends a pixel question to a text judge."
        )
    )
    binds: List[str] = Field(
        default_factory=list,
        description=(
            "Placeholder names this criterion's description accepts, e.g. "
            "['subject'] for a description containing '{subject}'. A seed supplies "
            "values through SeedCriterion.bind and the judge prompt substitutes "
            "them, so one shared criterion can ask about each seed's own subject "
            "while keeping one id, one weight and one column in every report."
        ),
    )
    tags: List[str] = Field(
        default_factory=list,
        description=(
            "Tags from the library's `criterion_tags` vocabulary. The analytic axis "
            "that cuts across sections — what the check looks at, how much of the "
            "video it needs, what kind of failure it catches. Reporting and "
            "visualisation only: a tag never decides whether a seed carries the "
            "criterion."
        ),
    )
    requires_references: bool = Field(
        default=False,
        description=(
            "The criterion only means something for a seed carrying reference "
            "images. A seed with none should simply not name it; a seed that "
            "names it and whose images fail to load is passed without a model "
            "call, so a dataset problem does not score as a video defect."
        ),
    )

    def model_post_init(self, _context) -> None:
        declared, used = set(self.binds), placeholders_in(self.description)
        if declared != used:
            raise ValueError(
                f"criterion {self.id!r}: `binds` is {sorted(declared)} but the "
                f"description uses {sorted(used)}; they must match exactly. A "
                f"declared bind nothing substitutes is dead weight; an undeclared "
                f"`{{placeholder}}` would reach the judge as a literal brace."
            )


_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def placeholders_in(text: str) -> set:
    """The `{name}` placeholders a criterion description declares inline."""
    return set(_PLACEHOLDER.findall(text))


def bind_description(criterion: "RubricCriterion", bind: Dict[str, str]) -> str:
    """
    The criterion's description with this seed's bind values substituted in.

    Raises on an unbound placeholder rather than leaving `{subject}` in the text.
    The judge would otherwise ask the model a question with a literal brace in it
    and score whatever came back, which is a dataset bug wearing a video defect's
    clothes.
    """
    missing = sorted(placeholders_in(criterion.description) - set(bind))
    if missing:
        raise ValueError(
            f"criterion {criterion.id!r} has unbound placeholders {missing}; "
            f"the seed must supply them under `bind`"
        )
    out = criterion.description
    for name, value in bind.items():
        out = out.replace("{" + name + "}", value)
    return out


class SafetyCheck(BaseModel):
    """One safety veto check (binary: violation or not). Applied to every seed."""

    id: str = Field(description="Stable id, e.g. 'D1'")
    category: str = Field(description="Safety category, e.g. 'Violence & Gore'")
    description: str = Field(description="The question the judge must answer")
    veto: bool = Field(default=True, description="A violation vetoes the whole video")


class RubricSection(BaseModel):
    """
    One category in the library: a name, and the criteria that belong to it.

    Organisational only. A section groups criteria for reading and for reporting; it
    attaches nothing to anything. A criterion belongs to exactly one, so a report can
    say which category a check came from without qualification.
    """

    key: str = Field(description="Section key, e.g. 'general' or 'music'")
    name: str = Field(description="Human-readable name")
    description: str = Field(default="", description="What kind of check lives here")
    criteria: List[RubricCriterion] = Field(min_length=1)


class RubricLibrary(BaseModel):
    """
    Every criterion the benchmark knows, grouped into categories and tagged.

    `criteria` is the flat list, in section order, because that is what every
    consumer downstream of the dataset layer wants — the judge, the report,
    `veb-compare`. Sections and tags are the axes those consumers group by.

    Nothing here selects anything. The library is a catalogue: the seed builder picks
    from it per seed, grounded in that seed's brief, and the judge asks only what the
    seed ended up naming.
    """

    dimensions: List[Dimension] = Field(min_length=1)
    criterion_tags: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "The closed tag vocabulary, as group -> values. Groups are documentation "
            "and a grouping for reports; the namespace is flat, so a tag may appear "
            "in only one group."
        ),
    )
    sections: List[RubricSection] = Field(min_length=1)

    def model_post_init(self, _context) -> None:
        self._criteria = [c for s in self.sections for c in s.criteria]
        self._by_id = {c.id: c for c in self._criteria}
        if len(self._by_id) != len(self._criteria):
            seen, dupes = set(), set()
            for c in self._criteria:
                (dupes if c.id in seen else seen).add(c.id)
            raise ValueError(
                f"duplicate criterion ids in the rubric library: {sorted(dupes)}. "
                f"A criterion is defined by exactly one section."
            )
        self._section_of = {c.id: s.key for s in self.sections for c in s.criteria}

        keys = [s.key for s in self.sections]
        if len(set(keys)) != len(keys):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            raise ValueError(f"duplicate section keys: {dupes}")

        known = {d.key for d in self.dimensions}
        unknown = sorted({c.dimension for c in self._criteria} - known)
        if unknown:
            raise ValueError(f"criteria reference unknown dimensions: {unknown}")

        self._tag_group = {}
        for group, values in self.criterion_tags.items():
            for value in values:
                if value in self._tag_group:
                    raise ValueError(
                        f"tag {value!r} is declared in two groups "
                        f"({self._tag_group[value]!r} and {group!r}); the namespace is flat"
                    )
                self._tag_group[value] = group
        if self._tag_group:
            # Only enforced when a vocabulary is declared, so a minimal library built
            # in a test does not have to carry one.
            bad = sorted({t for c in self._criteria for t in c.tags} - set(self._tag_group))
            if bad:
                raise ValueError(
                    f"criteria carry tags outside `criterion_tags`: {bad}. The "
                    f"vocabulary is closed so that a tag cannot silently split one "
                    f"report column into two."
                )

    @property
    def criteria(self) -> List[RubricCriterion]:
        """Every criterion, flattened in section order."""
        return self._criteria

    @property
    def tag_vocabulary(self) -> List[str]:
        """Every declared tag, flat, in declaration order."""
        return [v for values in self.criterion_tags.values() for v in values]

    def get(self, criterion_id: str) -> RubricCriterion:
        return self._by_id[criterion_id]

    def __contains__(self, criterion_id: object) -> bool:
        return criterion_id in self._by_id

    def section_of(self, criterion_id: str) -> str:
        """The key of the section that defines this criterion."""
        return self._section_of[criterion_id]

    def group_of(self, tag: str) -> str:
        """Which vocabulary group a tag belongs to, or '' if undeclared."""
        return self._tag_group.get(tag, "")

    def with_tag(self, tag: str) -> List[RubricCriterion]:
        """Every criterion carrying this tag, in library order."""
        return [c for c in self._criteria if tag in c.tags]

    def select(self, criterion_ids: List[str]) -> List[RubricCriterion]:
        """
        The named criteria, in the library's declared order rather than the
        seed's.

        Order is the library's on purpose: a seed lists ids in whatever order
        they were added, and two seeds sharing criteria should still produce
        report tables that read the same way down the page.
        """
        wanted = set(criterion_ids)
        return [c for c in self._criteria if c.id in wanted]

    def dimension_name(self, key: str) -> str:
        for d in self.dimensions:
            if d.key == key:
                return d.name
        return key


class TagFacet(BaseModel):
    """One analytic dimension a seed can be tagged on, with its allowed values."""

    key: str = Field(description="Facet key, e.g. 'editing_style'")
    values: List[str] = Field(min_length=1, description="The values this facet accepts")
    description: str = ""


class TagVocabulary(BaseModel):
    """
    The closed set of facets and values a dataset's seeds may be tagged with.

    Closed on purpose. Tags exist to be grouped by across hundreds of seeds, and a
    free-text tagger asked to describe editing style will write "talking head",
    "talking-head" and "talking_head" across one corpus and silently report them as
    three different populations. A value outside the vocabulary fails the load.
    """

    facets: List[TagFacet] = Field(default_factory=list)
    multi: List[str] = Field(
        default_factory=list,
        description="Facet keys that may carry more than one value; the rest are single",
    )

    def model_post_init(self, _context) -> None:
        self._values = {f.key: set(f.values) for f in self.facets}
        unknown = sorted(set(self.multi) - set(self._values))
        if unknown:
            raise ValueError(f"`multi` names facets that do not exist: {unknown}")

    @property
    def keys(self) -> List[str]:
        return [f.key for f in self.facets]

    def problems(self, tags: Dict[str, List[str]]) -> List[str]:
        """Every way a seed's tags violate the vocabulary. Empty when it is clean."""
        out: List[str] = []
        for key, values in tags.items():
            if key not in self._values:
                out.append(f"unknown tag facet {key!r} (known: {self.keys})")
                continue
            bad = sorted(set(values) - self._values[key])
            if bad:
                out.append(
                    f"tag {key!r} has values {bad} outside its vocabulary "
                    f"{sorted(self._values[key])}"
                )
            if len(values) > 1 and key not in self.multi:
                out.append(f"tag {key!r} is single-valued but got {values}")
        return out


@dataclass
class Dataset:
    """The full benchmark dataset: the rubric library + safety checks + seeds."""

    rubrics: RubricLibrary
    genres: Dict[str, str]
    safety_checks: List[SafetyCheck]
    seeds: List[Seed]
    tags: Optional[TagVocabulary] = None

    def criteria_for(self, seed: Seed) -> List[RubricCriterion]:
        """
        The criteria this seed is judged on, in library order, with each seed's
        bind values already substituted into the description.

        Covers both sources: the library criteria the seed names, and any
        `local_criteria` the seed defines itself.

        Only `scored` criteria are returned. A seed built by the seed builder keeps
        every criterion it ever proposed, verdicts attached, so the dataset stays a
        readable record of the build — but the judge grades what the policy admitted,
        and nothing else reaches it.

        Substitution happens here rather than in the judge so that a criterion is
        already a plain `RubricCriterion` by the time it leaves the dataset layer:
        every consumer downstream — prompt building, scoring, the HTML report — goes
        on treating `description` as the question to ask.
        """
        bindings = {c.id: c.bind for c in seed.rubrics if c.scored}
        local = seed.local_criteria_by_id()
        # Library criteria first, in library order, then the seed's own. Local ones
        # trail deliberately: a report column that only one seed has should not push
        # the shared columns around.
        chosen = self.rubrics.select([cid for cid in bindings if cid not in local])
        chosen += [local[cid] for cid in bindings if cid in local]
        return [
            criterion.model_copy(
                update={"description": bind_description(criterion, bindings[criterion.id])}
            )
            for criterion in chosen
        ]

    def genre_name(self, key: str) -> str:
        return self.genres.get(key, key)
