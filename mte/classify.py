"""Classify preprocessed lines into an intermediate representation label set.

Labels:
    part-marker      "PART ONE" style short letter-spaced-caps banner
    part-title       the title that follows a part-marker
    chapter-caption  short "Chapter N" or "C1" style label near a chapter title
    chapter-title    the chapter's headline
    h2               section heading below chapter
    h3               subsection heading
    body             normal paragraph text
    pull-quote       centered italic body-sized-ish sentence
    scene-break      decorative divider line
    cover-title      pre-TOC cover / title-page giant text
    caption          small centered label near a heading

The classifier only emits a per-line label + confidence. Grouping into
chapter sections, list detection, and table detection happen in a later
pass — this keeps the classifier easy to reason about and audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pdf_extract import Line
from .preprocess import is_decorative_only
from .tiers import TierAssignment


PART_RE = re.compile(r"^\s*PART\s+([IVXLCM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b", re.IGNORECASE)
CHAPTER_CAPTION_RE = re.compile(r"^\s*(CHAPTER\s+\d+|CH\.?\s*\d+|C\d+|\d+)\s*$", re.IGNORECASE)


@dataclass
class ClassifiedLine:
    line: Line
    label: str
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.line.text.strip()


def _mostly_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) > 0.85


def _looks_like_part_marker(text: str) -> bool:
    return bool(PART_RE.match(text.strip()))


def _looks_like_chapter_caption(text: str) -> bool:
    return bool(CHAPTER_CAPTION_RE.match(text.strip()))


def classify(lines: list[Line], tiers: TierAssignment) -> list[ClassifiedLine]:
    out: list[ClassifiedLine] = []
    body = tiers.body

    for ln in lines:
        text = ln.text.strip()
        if not text:
            continue
        size = ln.dominant_size
        role = tiers.role_for(size)
        centered = ln.is_centered
        bold = ln.bold_ratio > 0.5
        italic = ln.italic_ratio > 0.5
        caps = _mostly_caps(text)

        if is_decorative_only(text):
            out.append(ClassifiedLine(ln, "scene-break", 1.0))
            continue

        if role == "cover-title":
            out.append(ClassifiedLine(ln, "cover-title", 1.0))
            continue

        if _looks_like_part_marker(text) and centered and (bold or caps):
            out.append(ClassifiedLine(ln, "part-marker", 1.0, ["matched PART regex"]))
            continue

        if abs(size - body) <= 2.5 and italic and centered:
            out.append(ClassifiedLine(ln, "pull-quote", 0.9))
            continue

        if _looks_like_chapter_caption(text) and centered and (bold or caps) and size <= body + 4.0:
            out.append(ClassifiedLine(ln, "chapter-caption", 0.9))
            continue

        chapter_gate = centered and (bold or caps)
        if chapter_gate and role in ("part-title", "chapter"):
            label = "part-title" if role == "part-title" else "chapter-title"
            out.append(ClassifiedLine(ln, label, 1.0))
            continue

        h2h3_gate = bold or caps
        if h2h3_gate and role in ("h2", "h3"):
            out.append(ClassifiedLine(ln, role, 1.0))
            continue

        if role in ("part-title", "chapter", "h2", "h3"):
            out.append(ClassifiedLine(
                ln, "body", 0.5,
                [f"heading-tier size {size:.1f}pt but weak heading signals — kept as body"],
            ))
            continue

        if role == "pull-quote-tier" and italic and centered:
            out.append(ClassifiedLine(ln, "pull-quote", 1.0))
            continue

        if abs(size - body) <= 2.5 and bold and centered and caps:
            out.append(ClassifiedLine(ln, "body", 1.0, ["bold-caps centered body-sized punchline"]))
            continue

        out.append(ClassifiedLine(ln, "body", 1.0))

    _merge_split_chapter_captions(out, body)
    _demote_punchline_headings(out)
    _pair_chapter_captions(out, body)
    _pair_part_markers(out)
    _merge_wrapped_headings(out)
    _mark_front_matter(out)
    return out


_CHAPTER_C_NUM_RE = re.compile(r"^\s*C\s*(\d+)\s*$", re.IGNORECASE)
_HAPTER_TAIL_RE = re.compile(r"^\s*HAPTER\s*$", re.IGNORECASE)


def _merge_split_chapter_captions(items: list[ClassifiedLine], body_size: float) -> None:
    """Fuse "C<n>" caption line + adjacent "HAPTER" fragment into "CHAPTER <n>".

    Some manuscripts render the chapter caption with an oversized initial "C"
    and its number on one visual line, and "HAPTER" as a smaller fragment on
    an adjacent line. We normalize the caption text so it reads "CHAPTER n"
    and remove the orphaned fragment, which otherwise pollutes downstream
    caption→title pairing.
    """
    i = 0
    while i < len(items):
        ci = items[i]
        m = _CHAPTER_C_NUM_RE.match(ci.text) if ci.label == "chapter-caption" else None
        if m:
            num = m.group(1)
            for j in range(max(0, i - 2), min(len(items), i + 4)):
                if j == i:
                    continue
                cj = items[j]
                if cj.line.page != ci.line.page:
                    continue
                if _HAPTER_TAIL_RE.match(cj.text):
                    ci.line.spans[0].text = f"CHAPTER {num}"
                    del ci.line.spans[1:]
                    ci.notes.append("merged split 'C<n>' + 'HAPTER' fragment")
                    del items[j]
                    break
        i += 1


_TERMINAL_PUNCT = tuple(".!")


def _demote_punchline_headings(items: list[ClassifiedLine]) -> None:
    for c in items:
        if c.label not in ("h2", "h3"):
            continue
        text = c.text.rstrip()
        if not text.endswith(_TERMINAL_PUNCT):
            continue
        c.label = "body"
        c.confidence = 0.7
        c.notes.append("demoted from heading: ends with sentence terminator")


def _merge_wrapped_headings(items: list[ClassifiedLine]) -> None:
    merge_labels = {"chapter-title", "part-title", "h2", "h3"}
    i = 0
    while i < len(items) - 1:
        c = items[i]
        n = items[i + 1]
        if (
            c.label in merge_labels
            and n.label == c.label
            and c.line.page == n.line.page
            and abs(c.line.dominant_size - n.line.dominant_size) < 0.2
            and (n.line.y0 - c.line.y1) < c.line.dominant_size * 1.5
        ):
            joined = c.text.rstrip() + " " + n.text.lstrip()
            c.line.spans[0].text = joined
            del c.line.spans[1:]
            c.notes.append("merged wrapped heading")
            del items[i + 1]
            continue
        i += 1


def _mark_front_matter(items: list[ClassifiedLine]) -> None:
    first_part = next((i for i, c in enumerate(items) if c.label == "part-marker"), None)
    first_cap = next((i for i, c in enumerate(items) if c.label == "chapter-caption"), None)
    boundary = min([x for x in (first_part, first_cap) if x is not None], default=None)
    if boundary is None:
        return
    for c in items[:boundary]:
        if c.label in ("part-title", "chapter-title"):
            c.notes.append("in front-matter zone")


def _pair_chapter_captions(items: list[ClassifiedLine], body_size: float) -> None:
    for i, ci in enumerate(items):
        if ci.label != "chapter-caption":
            continue
        for j in range(i + 1, min(i + 6, len(items))):
            cj = items[j]
            if cj.label == "chapter-caption":
                break
            if cj.label in ("chapter-title", "h2", "h3"):
                cj.notes.append(f"paired with caption at index {i}")
                if cj.label != "chapter-title":
                    cj.label = "chapter-title"
                    cj.confidence = 0.8
                break
            if cj.label == "body":
                ln = cj.line
                if (
                    ln.dominant_size >= body_size - 0.5
                    and ln.bold_ratio > 0.5
                    and ln.is_centered
                    and _mostly_caps(cj.text)
                ):
                    cj.label = "chapter-title"
                    cj.confidence = 0.7
                    cj.notes.append(
                        f"promoted body→chapter-title (bold+caps+centered, follows caption at index {i})"
                    )
                    break


def _pair_part_markers(items: list[ClassifiedLine]) -> None:
    for i, ci in enumerate(items):
        if ci.label != "part-marker":
            continue
        for j in range(i + 1, min(i + 6, len(items))):
            cj = items[j]
            if cj.label == "part-marker":
                break
            if cj.label in ("part-title", "chapter-title", "h2", "h3"):
                if cj.label != "part-title":
                    cj.label = "part-title"
                    cj.confidence = 0.8
                    cj.notes.append(f"promoted to part-title (follows part-marker at index {i})")
                break


def outline_summary(items: list[ClassifiedLine]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in items:
        counts[c.label] = counts.get(c.label, 0) + 1
    return counts


def format_outline(items: list[ClassifiedLine], show_labels: set[str] | None = None) -> str:
    if show_labels is None:
        show_labels = {"part-marker", "part-title", "chapter-caption", "chapter-title", "h2", "h3"}
    indent_map = {
        "part-marker": "",
        "part-title": "  ",
        "chapter-caption": "    ",
        "chapter-title": "    ",
        "h2": "      ",
        "h3": "        ",
    }
    marker_map = {
        "part-marker": "▓",
        "part-title": "◆",
        "chapter-caption": "·",
        "chapter-title": "▸",
        "h2": "•",
        "h3": "∙",
    }
    lines_out: list[str] = []
    for c in items:
        if c.label not in show_labels:
            continue
        indent = indent_map.get(c.label, "")
        marker = marker_map.get(c.label, "-")
        text = c.text if len(c.text) <= 90 else c.text[:87] + "..."
        conf = f" ({c.confidence:.1f})" if c.confidence < 1.0 else ""
        lines_out.append(
            f"{indent}{marker} [p{c.line.page:>3} {c.line.dominant_size:>4.1f}pt {c.label:>15}{conf}]  {text}"
        )
    return "\n".join(lines_out)
