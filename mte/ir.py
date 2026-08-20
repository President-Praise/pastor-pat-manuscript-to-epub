"""Intermediate representation for a manuscript.

Types are strict: invalid states are unrepresentable. Each block type maps
1:1 to an HTML tag in the writer. Construction-time invariants catch bugs
in the classifier→IR converter before they reach serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Union


class IRConstructionError(ValueError):
    pass


@dataclass
class TextRun:
    text: str
    bold: bool = False
    italic: bool = False
    small_caps: bool = False
    link: str | None = None


@dataclass
class Paragraph:
    runs: list[TextRun]
    align: Literal["left", "center", "right"] = "left"
    drop_cap: bool = False

    def __post_init__(self) -> None:
        if not self.runs:
            raise IRConstructionError("Paragraph.runs must be non-empty")


@dataclass
class Heading:
    level: Literal[2, 3]
    runs: list[TextRun]

    def __post_init__(self) -> None:
        if self.level not in (2, 3):
            raise IRConstructionError(f"Heading.level must be 2 or 3, got {self.level}")
        if not self.runs:
            raise IRConstructionError("Heading.runs must be non-empty")


PullQuoteRole = Literal["plain", "subtitle-tagline", "scripture", "epigraph"]


@dataclass
class PullQuote:
    runs: list[TextRun]
    attribution: str | None = None
    role: PullQuoteRole = "plain"

    def __post_init__(self) -> None:
        if not self.runs:
            raise IRConstructionError("PullQuote.runs must be non-empty")
        if self.role not in ("plain", "subtitle-tagline", "scripture", "epigraph"):
            raise IRConstructionError(f"PullQuote.role invalid: {self.role}")


@dataclass
class SceneBreak:
    pass


@dataclass
class ListItem:
    runs: list[TextRun]
    children: list["Block"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.runs:
            raise IRConstructionError("ListItem.runs must be non-empty")


@dataclass
class ItemSegment:
    items: list[ListItem]

    def __post_init__(self) -> None:
        if not self.items:
            raise IRConstructionError("ItemSegment.items must be non-empty")


@dataclass
class InterruptionSegment:
    blocks: list["Block"]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise IRConstructionError("InterruptionSegment.blocks must be non-empty")


ListSegment = Union[ItemSegment, InterruptionSegment]


@dataclass
class ListBlock:
    style: Literal["ordered", "unordered"]
    segments: list[ListSegment]
    start: int = 1

    def __post_init__(self) -> None:
        if self.style not in ("ordered", "unordered"):
            raise IRConstructionError(f"ListBlock.style invalid: {self.style}")
        if not self.segments:
            raise IRConstructionError("ListBlock.segments must be non-empty")
        if not any(isinstance(s, ItemSegment) for s in self.segments):
            raise IRConstructionError("ListBlock must contain at least one ItemSegment")

    def item_starts(self) -> list[int]:
        """Return the item-number that each ItemSegment starts at (1-based, offset
        by self.start). Interruption segments contribute nothing to numbering."""
        starts: list[int] = []
        running = self.start
        for seg in self.segments:
            if isinstance(seg, ItemSegment):
                starts.append(running)
                running += len(seg.items)
        return starts


@dataclass
class TableCell:
    blocks: list["Block"]
    colspan: int = 1
    rowspan: int = 1


@dataclass
class TableRow:
    cells: list[TableCell]

    def __post_init__(self) -> None:
        if not self.cells:
            raise IRConstructionError("TableRow.cells must be non-empty")


@dataclass
class Table:
    body_rows: list[TableRow]
    header_rows: list[TableRow] = field(default_factory=list)
    caption: str | None = None
    source_confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.body_rows:
            raise IRConstructionError("Table.body_rows must be non-empty")
        widths = {
            sum(c.colspan for c in row.cells)
            for row in self.body_rows + self.header_rows
        }
        if len(widths) > 1:
            raise IRConstructionError(
                f"Table rows have inconsistent widths: {widths}"
            )


@dataclass
class Callout:
    variant: Literal["scripture", "note", "warning", "sidebar"]
    content: list["Block"]

    def __post_init__(self) -> None:
        if self.variant not in ("scripture", "note", "warning", "sidebar"):
            raise IRConstructionError(f"Callout.variant invalid: {self.variant}")
        if not self.content:
            raise IRConstructionError("Callout.content must be non-empty")


Block = Union[
    Paragraph, Heading, PullQuote, SceneBreak, ListBlock, Table, Callout
]


@dataclass
class Section:
    title: str
    kind: Literal["chapter", "front", "back", "cover"]
    slug: str
    content: list[Block] = field(default_factory=list)
    caption: str | None = None

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise IRConstructionError("Section.title must be non-empty")
        if self.kind not in ("chapter", "front", "back", "cover"):
            raise IRConstructionError(f"Section.kind invalid: {self.kind}")
        if not self.slug.strip():
            raise IRConstructionError("Section.slug must be non-empty")


@dataclass
class Part:
    marker: str
    title: str
    chapters: list[Section]
    slug: str
    intro: list[Block] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.marker.strip():
            raise IRConstructionError("Part.marker must be non-empty")
        if not self.title.strip():
            raise IRConstructionError("Part.title must be non-empty")
        if not self.chapters:
            raise IRConstructionError("Part.chapters must be non-empty")


@dataclass
class BookMetadata:
    title: str
    author: str
    language: str = "en"
    identifier: str | None = None

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise IRConstructionError("BookMetadata.title must be non-empty")
        if not self.author.strip():
            raise IRConstructionError("BookMetadata.author must be non-empty")


@dataclass
class Book:
    metadata: BookMetadata
    front_matter: list[Section] = field(default_factory=list)
    parts: list[Part] = field(default_factory=list)
    chapters: list[Section] = field(default_factory=list)
    back_matter: list[Section] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.parts and self.chapters:
            raise IRConstructionError(
                "Book has both parts and top-level chapters; put chapters inside parts"
            )

    def walk_sections(self) -> Iterator[Section]:
        yield from self.front_matter
        for part in self.parts:
            yield from part.chapters
        yield from self.chapters
        yield from self.back_matter
