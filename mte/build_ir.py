"""Convert ClassifiedLine[] into a Book IR.

Passes:
    1. Walk classified lines; segment into sections at chapter/part boundaries.
    2. Within each section, convert lines to Blocks (with list detection).
    3. Group sections into front_matter / parts / back_matter.
    4. Assign stable, unique slugs across the book.

The classifier tells us WHAT each line is; this module decides WHERE each
section belongs and reconstructs list continuity across callout interruptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .classify import ClassifiedLine
from .ir import (
    Block, Book, BookMetadata, Heading, InterruptionSegment, ItemSegment,
    ListBlock, ListItem, Paragraph, Part, PullQuote, SceneBreak, Section,
    Table, TableCell, TableRow, TextRun,
)


_BACK_MATTER_KEYWORDS = (
    "APPENDIX", "APPENDICES", "GLOSSARY", "INDEX", "EPILOGUE", "AFTERWORD",
    "BIBLIOGRAPHY", "REFERENCES", "FINAL WORD", "ABOUT THE AUTHOR",
)

# Numbered/lettered list marker. Body normally has a space after "1." / "a)"
# but PDFs sometimes render it without one ("28.The Lord..."). We accept the
# missing-space form only when followed by an uppercase letter (or quote/
# open-bracket) to avoid catching decimals like "2.5" as bogus list items.
_ORDERED_ITEM_RE = re.compile(
    r"^\s*\(?(\d{1,3}|[a-zA-Z])[\)\.\:]"
    r"(?:\s+|(?=[\"“'‘A-Z]))"
    r"(.+)$",
    re.DOTALL,
)
_BULLET_CHARS = "•●◆◇◈❖❦♚♛✧✩✪★☆❧⚜❈❃❊*·"
# Interior bullets used to split a single visual line that packs multiple
# bullet items ("• Foo • Bar • Baz"). Narrower than _BULLET_CHARS because
# `*` and `·` show up mid-word in body text and would over-split.
_INTERIOR_BULLET_CHARS = "•●◆◇◈❖❦❈❃❊"
_INTERIOR_BULLET_RE = re.compile(r"\s*[" + re.escape(_INTERIOR_BULLET_CHARS) + r"]\s*")
# Body normally has a space after "•" but PDFs sometimes render it without
# ("•The Lord..."). Accept the missing-space form only when followed by a
# letter or opening quote/bracket — the same fix already applied to
# _ORDERED_ITEM_RE for "28.The" style numbered items.
_BULLET_RE = re.compile(
    r"^\s*([" + re.escape(_BULLET_CHARS) + r"])"
    r"(?:\s+|(?=[\"“'‘A-Za-z0-9\(\[]))"
    r"(.+)$",
    re.DOTALL,
)

# Scripture heuristics — either a bracketed translation abbreviation, or a
# "— Book chap:verse" style citation.
_SCRIPTURE_TRANS_RE = re.compile(
    r"\((?:NKJV|NIV|ESV|KJV|AMP|AMPC|NLT|NASB|GNT|MSG|CSB|NRSV|RSV|ASV|NCV|CEV|TLB|TPT|WEB|HCSB|YLT|DBY|GNV|BBE)\)",
    re.IGNORECASE,
)
_SCRIPTURE_CITE_RE = re.compile(
    r"[—–\-]\s*(?:[1-3]\s+)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s+\d+:\d+"
)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "section"


def _make_slugs_unique(desired: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for s in desired:
        n = seen.get(s, 0) + 1
        seen[s] = n
        out.append(s if n == 1 else f"{s}-{n}")
    return out


def _make_run(cl: ClassifiedLine) -> TextRun:
    ln = cl.line
    return TextRun(
        text=cl.text.strip(),
        bold=ln.bold_ratio > 0.5,
        italic=ln.italic_ratio > 0.5,
    )


def _make_runs_from_spans(cl: ClassifiedLine) -> list[TextRun]:
    """Convert one line into TextRuns grouped by (bold, italic) span style.

    Per-span grouping preserves within-line style boundaries a line-level
    ratio would flatten — critical for lines that pack an italic quote tail
    and a bold scripture citation onto the same visual row. Falls back to a
    line-level run when the line has no span detail.
    """
    ln = cl.line
    if not ln.spans:
        text = cl.text.strip()
        if not text:
            return []
        return [TextRun(
            text=text,
            bold=ln.bold_ratio > 0.5,
            italic=ln.italic_ratio > 0.5,
        )]
    runs: list[TextRun] = []
    for span in ln.spans:
        text = span.text
        if not text:
            continue
        if runs and runs[-1].bold == span.bold and runs[-1].italic == span.italic:
            runs[-1].text += text
        else:
            runs.append(TextRun(text=text, bold=span.bold, italic=span.italic))
    if runs:
        runs[0].text = runs[0].text.lstrip()
        runs[-1].text = runs[-1].text.rstrip()
        runs = [r for r in runs if r.text]
    return runs


def _run(text: str, bold: bool = False, italic: bool = False) -> TextRun:
    return TextRun(text=text, bold=bold, italic=italic)


def _detect_list_item(text: str) -> tuple[str, str] | None:
    if not text:
        return None
    if text[0] in _BULLET_CHARS:
        m = _BULLET_RE.match(text)
        if m:
            return ("unordered", m.group(2).strip())
    m = _ORDERED_ITEM_RE.match(text)
    if m:
        return ("ordered", m.group(2).strip())
    return None


def _paragraph_align(cl: ClassifiedLine) -> str:
    ln = cl.line
    if not ln.is_centered:
        return "left"
    # A "centered" single line that spans most of the page width is really
    # body text — fully-justified wrap lines have symmetric margins by
    # definition. Same guard as _paragraph_align_multi.
    if (ln.x1 - ln.x0) > ln.page_width * 0.70:
        return "left"
    return "center"


def _paragraph_align_multi(cls: list[ClassifiedLine]) -> str:
    """Alignment for a paragraph assembled from multiple lines.

    A run of lines whose `is_centered` is True can still be justified body
    text (fully-justified wrap lines have symmetric margins by definition).
    Treat the paragraph as centered only when every line's rendered width is
    well short of the page width — the hallmark of true centering.
    """
    for cl in cls:
        ln = cl.line
        if not ln.is_centered:
            return "left"
        if (ln.x1 - ln.x0) > ln.page_width * 0.70:
            return "left"
    return "center"


def _is_back_matter(title: str) -> bool:
    upper = title.strip().upper()
    return any(kw in upper for kw in _BACK_MATTER_KEYWORDS)


# --------------------------------------------------------------------------- #
# Section segmentation
# --------------------------------------------------------------------------- #

@dataclass
class _FlatSection:
    kind: str                     # "front", "chapter", "back", "cover"
    caption: str | None
    title: str
    body_lines: list[ClassifiedLine] = field(default_factory=list)
    is_cover: bool = False
    cover_start_page: int | None = None


@dataclass
class _PartRecord:
    marker: str
    title: str
    chapters: list[_FlatSection] = field(default_factory=list)
    intro_lines: list[ClassifiedLine] = field(default_factory=list)


def _segment_sections(items: list[ClassifiedLine]) -> tuple[
    list[_FlatSection], list[_PartRecord], list[_FlatSection], list[_FlatSection]
]:
    """Walk classified lines, emitting front matter, parts (each with chapter
    sections), loose top-level chapters, and back matter."""

    front: list[_FlatSection] = []
    parts: list[_PartRecord] = []
    loose_chapters: list[_FlatSection] = []
    back: list[_FlatSection] = []

    # If the book has any part-titles at all, anything before the first
    # part-title stays as front matter — otherwise a stray front-matter
    # section labeled as chapter-caption would strand real front matter.
    has_parts = any(cl.label == "part-title" for cl in items)

    current_section: _FlatSection | None = None
    current_part: _PartRecord | None = None
    pending_part_marker: str | None = None
    pending_caption: str | None = None
    state = "front"

    def _place(sec: _FlatSection) -> None:
        nonlocal state
        if _is_back_matter(sec.title) and state in ("in_part", "in_chapters"):
            state = "back"
            sec.kind = "back"
            back.append(sec)
            return
        if state == "back":
            sec.kind = "back"
            back.append(sec)
        elif state == "front":
            sec.kind = "front"
            front.append(sec)
        elif state == "in_part" and current_part is not None:
            sec.kind = "chapter"
            current_part.chapters.append(sec)
        elif state == "in_chapters":
            sec.kind = "chapter"
            loose_chapters.append(sec)
        else:
            sec.kind = "back"
            back.append(sec)

    def _close_current() -> None:
        nonlocal current_section
        if current_section is not None:
            _place(current_section)
            current_section = None

    for cl in items:
        label = cl.label
        text = cl.text.strip()

        # Any title-like line on the same page as the initial cover-title is
        # title-page decoration (subtitle, author name) — fold in as content
        # rather than opening a new section or a spurious Part.
        if (
            current_section is not None
            and current_section.is_cover
            and current_section.cover_start_page == cl.line.page
            and label in ("cover-title", "chapter-title", "part-title", "part-marker")
        ):
            current_section.body_lines.append(cl)
            continue

        # A heading on a later page of a cover section marks the copyright /
        # dedication page — split it off so the cover is its own spine entry
        # and the copyright starts fresh, using the heading text as the new
        # section's title.
        if (
            current_section is not None
            and current_section.is_cover
            and cl.line.page != current_section.cover_start_page
            and label in ("h2", "h3")
            and text
        ):
            _close_current()
            current_section = _FlatSection(
                kind="front", caption=None, title=text,
            )
            continue

        if label == "part-marker":
            _close_current()
            pending_part_marker = text
            continue

        if label == "part-title":
            _close_current()
            marker = pending_part_marker or ""
            pending_part_marker = None
            if not marker:
                marker = f"PART {len(parts) + 1}"
            current_part = _PartRecord(marker=marker, title=text)
            parts.append(current_part)
            state = "in_part"
            continue

        if label == "chapter-caption":
            _close_current()
            if state == "front" and not has_parts:
                state = "in_chapters"
            pending_caption = text
            continue

        if label == "chapter-title":
            _close_current()
            if state == "front" and not has_parts:
                state = "in_chapters"
            sec_kind = "chapter" if state in ("in_part", "in_chapters") else state
            current_section = _FlatSection(
                kind=sec_kind,
                caption=pending_caption,
                title=text,
            )
            pending_caption = None
            continue

        if label == "cover-title":
            _close_current()
            current_section = _FlatSection(
                kind="front",
                caption=None,
                title=text,
                is_cover=True,
                cover_start_page=cl.line.page,
            )
            continue

        # Non-boundary line: append to current section if any, else buffer
        # into the Part's intro (if we're in a part before any chapter has
        # opened) or a synthetic "untitled preface" section.
        if current_section is None:
            if not text and label != "scene-break":
                continue
            if (
                state == "in_part"
                and current_part is not None
                and not current_part.chapters
            ):
                current_part.intro_lines.append(cl)
                continue
            kind_for_orphan = "front" if state in ("front", "in_part", "in_chapters") else state
            current_section = _FlatSection(
                kind=kind_for_orphan,
                caption=None,
                title="Preface",
            )
        current_section.body_lines.append(cl)

    _close_current()

    if pending_part_marker and current_part is None:
        # A bare part-marker with no title — synthesize a placeholder title so
        # the part is representable.
        parts.append(_PartRecord(marker=pending_part_marker, title=pending_part_marker.title()))

    return front, parts, loose_chapters, back


# --------------------------------------------------------------------------- #
# Body → Block conversion (with list detection)
# --------------------------------------------------------------------------- #

_HEADING_LEVEL = {"h2": 2, "h3": 3}


def _line_to_simple_block(cl: ClassifiedLine) -> Block | None:
    label = cl.label
    text = cl.text.strip()
    if not text and label != "scene-break":
        return None
    if label == "scene-break":
        return SceneBreak()
    runs = _make_runs_from_spans(cl) or [_make_run(cl)]
    if label == "pull-quote":
        return PullQuote(runs=runs)
    if label in ("h2", "h3"):
        return Heading(level=_HEADING_LEVEL[label], runs=runs)
    return Paragraph(runs=runs, align=_paragraph_align(cl))


def _convert_body_to_blocks(
    lines: list[ClassifiedLine], *, join_paragraphs: bool = True,
) -> list[Block]:
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        cl = lines[i]
        text = cl.text.strip()

        if cl.label == "body":
            table, consumed = _try_collect_table(lines, i)
            if table is not None:
                blocks.append(table)
                i += consumed
                continue

            match = _detect_list_item(text)
            if match:
                list_block, consumed = _collect_list(lines, i, match[0])
                if list_block is not None:
                    blocks.append(list_block)
                    i += consumed
                    continue

            if join_paragraphs:
                # Collect consecutive body lines into one paragraph unless
                # there's a positional signal (vertical gap, first-line indent,
                # font-size change).
                para_cls, consumed = _collect_paragraph(lines, i)
                if para_cls:
                    blocks.append(_build_paragraph(para_cls))
                    i += consumed
                    continue

        block = _line_to_simple_block(cl)
        if block is not None:
            blocks.append(block)
        i += 1

    return blocks


_TOC_TITLE_PATTERNS = ("TABLE OF CONTENTS", "CONTENTS", "TOC")


def _looks_like_toc_title(title: str) -> bool:
    t = title.strip().upper()
    return t in _TOC_TITLE_PATTERNS


def _content_has_substance(blocks: list[Block]) -> bool:
    """True iff the block list has anything besides scene-breaks."""
    return any(not isinstance(b, SceneBreak) for b in blocks)


_SYNTHETIC_PREFACE_TITLE = "Preface"


def _visible_text_len(blocks: list[Block]) -> int:
    total = 0
    for b in blocks:
        if isinstance(b, (Paragraph, Heading, PullQuote)):
            total += sum(len(r.text) for r in b.runs)
        elif isinstance(b, ListBlock):
            for seg in b.segments:
                if isinstance(seg, ItemSegment):
                    for it in seg.items:
                        total += sum(len(r.text) for r in it.runs)
    return total


def _is_meaningful_section(sec: Section) -> bool:
    """Filter for sections worth keeping in the final book.

    Drops synthetic-titled "Preface" stubs with trivial content — those
    arise when a stray body fragment (e.g. a leaked page number "i") gets
    orphaned before any real title and would otherwise render as a
    misleading empty-looking preface page.
    """
    if not _content_has_substance(sec.content):
        return False
    if sec.title == _SYNTHETIC_PREFACE_TITLE and _visible_text_len(sec.content) < 20:
        return False
    return True


def _is_paragraph_break(prev_cl: ClassifiedLine, next_cl: ClassifiedLine) -> bool:
    """True when visual signals say prev and next belong to different paragraphs.

    Alignment is deliberately NOT a break signal: fully-justified body wraps
    show `is_centered=True` (symmetric margins) while the paragraph's ragged
    last line does not — that flip is layout artifact, not paragraph boundary.
    """
    prev_ln = prev_cl.line
    next_ln = next_cl.line

    if abs(prev_ln.dominant_size - next_ln.dominant_size) > 1.0:
        return True

    # Bold-run-in-heading signal: a body line that is (almost) entirely bold
    # sitting adjacent to a body line that is (almost) entirely non-bold is a
    # subheading / body-prose boundary, not a wrapped paragraph. Without this
    # signal, a run-in subheading like "We may pray God's promises" glues to
    # the following prose in a single <p>, and — worse, across page breaks —
    # the preceding sentence's tail gets fused with the next page's bold
    # heading. Threshold pair (≥0.85 vs <0.5) is picked to fire on true
    # bold headings while ignoring paragraphs that merely contain a bold
    # phrase mid-sentence.
    prev_all_bold = prev_ln.bold_ratio >= 0.85
    next_all_bold = next_ln.bold_ratio >= 0.85
    prev_body_weight = prev_ln.bold_ratio < 0.5
    next_body_weight = next_ln.bold_ratio < 0.5
    if (prev_all_bold and next_body_weight) or (next_all_bold and prev_body_weight):
        return True

    # Same-page blank-line signal.
    if prev_ln.page == next_ln.page:
        gap = next_ln.y0 - prev_ln.y1
        line_h = max(prev_ln.dominant_size, next_ln.dominant_size)
        if gap > line_h * 0.6:
            return True

    # First-line indent: next starts noticeably right of the previous line.
    # Only meaningful for left-aligned text — a centered line's x0 shifts with
    # its length, so comparing x0 between centered lines would mis-classify a
    # short wrap-line as a new paragraph. Skip the check if either line is
    # centered.
    if not (prev_ln.is_centered or next_ln.is_centered):
        indent_em = next_ln.dominant_size * 0.7
        if next_ln.x0 - prev_ln.x0 > indent_em:
            return True

    return False


def _collect_paragraph(
    lines: list[ClassifiedLine], start: int,
) -> tuple[list[ClassifiedLine], int]:
    para: list[ClassifiedLine] = [lines[start]]
    i = start + 1
    while i < len(lines):
        nxt = lines[i]
        if nxt.label != "body":
            break
        if _detect_list_item(nxt.text.strip()):
            break
        if _is_paragraph_break(para[-1], nxt):
            break
        para.append(nxt)
        i += 1
    return para, i - start


_CONTINUATION_JOIN = " "


def _build_paragraph(cls: list[ClassifiedLine]) -> Paragraph:
    """Fuse a run of body ClassifiedLines into one Paragraph.

    Uses per-span (bold, italic) so mixed-style lines produce distinct runs
    — the italic tail of a quote and the bold citation that follows keep
    their semantic distinction instead of being collapsed by line-level
    ratio. Strips soft-hyphenation at line ends when the next line begins
    lowercase.
    """
    runs: list[TextRun] = []
    for line_idx, cl in enumerate(cls):
        line_runs = _make_runs_from_spans(cl)
        if not line_runs:
            continue
        for run_idx, r in enumerate(line_runs):
            is_line_start = (run_idx == 0 and line_idx > 0 and bool(runs))
            same_style = (
                bool(runs)
                and runs[-1].bold == r.bold
                and runs[-1].italic == r.italic
            )
            if same_style:
                prev = runs[-1]
                if is_line_start:
                    if prev.text.endswith("-") and r.text and r.text[0].islower():
                        prev.text = prev.text[:-1] + r.text
                    elif prev.text.endswith(("–", "—")):
                        prev.text = prev.text + r.text
                    else:
                        prev.text = prev.text + _CONTINUATION_JOIN + r.text
                else:
                    prev.text = prev.text + r.text
            else:
                text = r.text
                if is_line_start:
                    prev = runs[-1]
                    if prev.text.endswith("-") and text and text[0].islower():
                        prev.text = prev.text[:-1]
                    elif not prev.text.endswith(("–", "—", " ")) and not text.startswith(" "):
                        text = _CONTINUATION_JOIN + text
                runs.append(TextRun(text=text, bold=r.bold, italic=r.italic))
    if not runs:
        for cl in cls:
            t = cl.text.strip()
            if t:
                runs.append(TextRun(text=t))
                break
    return Paragraph(runs=runs, align=_paragraph_align_multi(cls))


# --------------------------------------------------------------------------- #
# Table detection
# --------------------------------------------------------------------------- #

# A gap between adjacent spans wider than this is treated as a column
# separator rather than a normal inter-word space. Body text spaces are ~3pt
# at 10pt Georgia; column gutters in the tables run 15–60pt.
_COLUMN_GAP_THRESHOLD_PT = 12.0

# Column-band matching tolerance (either edge may drift by this much across
# rows before we consider it a different band).
_COLUMN_BAND_TOL_PT = 8.0

# Minimum consecutive columnar lines required to recognize a table.
_MIN_TABLE_LINES = 2


def _line_column_groups(cl: ClassifiedLine) -> list[tuple[float, float, str]]:
    """Group a line's spans into columns by horizontal gap.

    Returns [(x0, x1, text), ...] with whitespace-only spans dropped.
    Returns [] if the line has fewer than 2 non-empty columns.
    """
    spans = [s for s in cl.line.spans if s.text.strip()]
    if len(spans) < 2:
        return []
    spans = sorted(spans, key=lambda s: s.x0)
    groups: list[list] = [[spans[0]]]
    for s in spans[1:]:
        prev_x1 = groups[-1][-1].x1
        if s.x0 - prev_x1 >= _COLUMN_GAP_THRESHOLD_PT:
            groups.append([s])
        else:
            groups[-1].append(s)
    if len(groups) < 2:
        return []
    out: list[tuple[float, float, str]] = []
    for g in groups:
        text = "".join(s.text for s in g).strip()
        if not text:
            continue
        out.append((g[0].x0, g[-1].x1, text))
    if len(out) < 2:
        return []
    # Reject lines whose first "column" is really a list marker (bullet or
    # ordered), which would otherwise look like a 2-column "marker | text" row.
    first_text = out[0][2]
    if first_text:
        if first_text[0] in _BULLET_CHARS and len(first_text) <= 2:
            return []
        if _ORDERED_ITEM_RE.match(first_text + (" x" if not first_text.endswith(" ") else "x")):
            return []
    return out


def _band_index(x0: float, x1: float, bands: list[tuple[float, float]]) -> int | None:
    """Return the band index whose x0 is closest to `x0`, if the column
    starts within tolerance of that band's left edge."""
    center = (x0 + x1) / 2.0
    best_i: int | None = None
    best_d = float("inf")
    for i, (bx0, bx1) in enumerate(bands):
        # Prefer left-edge alignment (columns are left-aligned in this
        # manuscript), but fall back to any band that contains our center.
        d_left = abs(x0 - bx0)
        if d_left < best_d:
            best_d = d_left
            best_i = i
    if best_i is None:
        return None
    bx0, bx1 = bands[best_i]
    if abs(x0 - bx0) <= _COLUMN_BAND_TOL_PT:
        return best_i
    # Fallback: center within extended band range.
    if bx0 - _COLUMN_BAND_TOL_PT <= center <= bx1 + _COLUMN_BAND_TOL_PT:
        return best_i
    return None


def _single_column_continuation(
    cl: ClassifiedLine,
    bands: list[tuple[float, float]],
    prev_row_cell_texts: list[str] | None = None,
) -> tuple[float, float, str] | None:
    """Detect a line that is a wrapped continuation of a cell.

    Non-first-column continuations (band index ≥ 1) are always accepted —
    they cannot be mistaken for a new row because a new row must start in
    column 0.

    First-column continuations (band index 0) are accepted only when the
    text obviously continues the previous row's band-0 cell (e.g.
    "2 Corinthians" wraps to "12:11"). This uses the shared wrap-continuation
    heuristic so table wrap detection stays consistent with list/pull-quote
    wrap detection.
    """
    spans = [s for s in cl.line.spans if s.text.strip()]
    if len(spans) != 1:
        return None
    s = spans[0]
    idx = _band_index(s.x0, s.x1, bands)
    if idx is None:
        return None
    text = s.text.strip()
    if idx == 0:
        if not prev_row_cell_texts or idx >= len(prev_row_cell_texts):
            return None
        prev_cell = prev_row_cell_texts[idx]
        if not prev_cell:
            return None
        if not _looks_like_wrap_continuation(prev_cell, text):
            return None
    return (s.x0, s.x1, text)


def _try_collect_table(
    lines: list[ClassifiedLine], start: int,
) -> tuple[Table | None, int]:
    """Detect a run of columnar body lines starting at lines[start].

    Rows are lines whose spans split into ≥2 columns separated by wide
    horizontal gaps. Consecutive rows must share the same column x-bands.
    Lines missing the first column are treated as continuations of the
    previous row (their cells append to the corresponding band).
    """
    if lines[start].label != "body":
        return None, 1
    first_cols = _line_column_groups(lines[start])
    if not first_cols:
        return None, 1

    bands: list[tuple[float, float]] = [(x0, x1) for (x0, x1, _t) in first_cols]

    n_cols = len(bands)

    def _current_row_cell_texts(rows: list[list[tuple[float, float, str]]]) -> list[str]:
        """Accumulate cell texts for the primary row currently in progress —
        starts at the last row whose column set covers band 0."""
        acc: list[str] = [""] * n_cols
        start_idx = 0
        for k in range(len(rows) - 1, -1, -1):
            if any(_band_index(x0, x1, bands) == 0 for (x0, x1, _t) in rows[k]):
                start_idx = k
                break
        for row in rows[start_idx:]:
            for (x0, x1, text) in row:
                idx = _band_index(x0, x1, bands)
                if idx is None:
                    continue
                sep = "" if acc[idx].endswith(("–", "—")) or not acc[idx] else " "
                acc[idx] = acc[idx] + sep + text if acc[idx] else text
        return acc

    # Rows tagged is_continuation=True are wrap-fragments of the previous
    # primary row and must NOT start a new group during row-grouping below —
    # otherwise a wrapped "12:11" tail of "2 Corinthians" would render as its
    # own table row instead of merging into the correct cell.
    raw_rows: list[tuple[list[tuple[float, float, str]], bool]] = [(first_cols, False)]
    i = start + 1
    while i < len(lines):
        cl = lines[i]
        if cl.label != "body":
            break
        cols = _line_column_groups(cl)
        if cols:
            indices = [_band_index(x0, x1, bands) for (x0, x1, _t) in cols]
            if any(idx is None for idx in indices):
                break
            raw_rows.append((cols, False))
            i += 1
            continue
        # Single-column continuation of an in-progress table: a body line
        # with one non-empty span whose x-position fits a band. Column 2+
        # continuations are unambiguous; column 0 continuations are accepted
        # only when the text is a wrap-continuation of the current primary
        # row's band-0 cell (e.g. "2 Corinthians" + "12:11").
        prev_texts = _current_row_cell_texts([r for (r, _) in raw_rows])
        single = _single_column_continuation(
            cl, bands, prev_row_cell_texts=prev_texts,
        )
        if single is not None:
            raw_rows.append(([single], True))
            i += 1
            continue
        break

    if len(raw_rows) < _MIN_TABLE_LINES:
        return None, 1

    # Group continuation lines (no content in column 0, OR flagged as a
    # wrap continuation of the current primary row) into the preceding
    # primary row.
    def _covers_first_band(cols_) -> bool:
        for (x0, x1, _t) in cols_:
            if _band_index(x0, x1, bands) == 0:
                return True
        return False

    grouped: list[list[list[tuple[float, float, str]]]] = []
    for (row, is_continuation) in raw_rows:
        starts_new = (_covers_first_band(row) and not is_continuation) or not grouped
        if starts_new:
            grouped.append([row])
        else:
            grouped[-1].append(row)

    table_rows: list[TableRow] = []
    for group in grouped:
        cell_texts: list[str] = [""] * n_cols
        for row in group:
            for (x0, x1, text) in row:
                idx = _band_index(x0, x1, bands)
                if idx is None:
                    continue
                if cell_texts[idx]:
                    sep = "" if cell_texts[idx].endswith(("–", "—")) else " "
                    cell_texts[idx] = cell_texts[idx] + sep + text
                else:
                    cell_texts[idx] = text
        cells: list[TableCell] = []
        for t in cell_texts:
            body_text = t if t else " "
            cells.append(TableCell(blocks=[Paragraph(runs=[TextRun(text=body_text)])]))
        table_rows.append(TableRow(cells=cells))

    # Promote a bold-heavy first row to the header.
    header_rows: list[TableRow] = []
    body_rows = table_rows
    if lines[start].line.bold_ratio > 0.5 and len(body_rows) > 1:
        header_rows = [body_rows.pop(0)]

    if not body_rows:
        return None, 1

    return Table(header_rows=header_rows, body_rows=body_rows), i - start


_CONTINUATION_LEADERS = (")", "]", "}", ",", ";", ":", ".", "?", "!", "”", "’", "\"", "'", "(")

# Sentence-terminating characters. When the previous fragment ends with one of
# these (optionally followed by a closing quote/bracket) it's a complete
# sentence and the next line is *not* a wrapped continuation.
_SENTENCE_TERMINATORS = (".", "!", "?")

# Words that, when they end a fragment, strongly signal the next line
# continues the sentence — the fragment is syntactically incomplete because
# the word demands an object (articles/prepositions/conjunctions/etc.) or
# introduces a numeric reference (chapter/verse/page).
_CONTINUATION_TAIL_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet",
    "of", "in", "on", "at", "by", "to", "for", "with", "from",
    "as", "is", "are", "was", "were", "be", "been", "being",
    "chapter", "chapters", "verse", "verses", "page", "pages",
    "section", "part", "book", "psalm", "psalms",
})


def _last_word(text: str) -> str:
    """Return the last whitespace-delimited word, stripped of trailing punctuation."""
    stripped = text.rstrip()
    if not stripped:
        return ""
    tokens = stripped.split()
    if not tokens:
        return ""
    return tokens[-1].strip(",;:()[]{}\"'“”‘’")


def _ends_with_sentence_terminator(text: str) -> bool:
    """True if text (after trailing quotes/brackets) ends with . ! or ?"""
    stripped = text.rstrip().rstrip("”’\")]}'\"")
    return bool(stripped) and stripped[-1] in _SENTENCE_TERMINATORS


def _looks_like_wrap_continuation(prev_text: str, next_text: str) -> bool:
    """True when `next_text` is the tail of `prev_text` split by a PDF line-wrap.

    Shared heuristic used by every container type (list items, pull-quotes,
    table cells) so the same class of PDF-fragmentation bug can't sneak in via
    one path and get missed by another.

    Signals (any triggers True):
      - next begins lowercase (mid-sentence wrap)
      - next opens with a closing bracket / quote / trailing punctuation
      - prev has an unmatched '(' (citation being completed on next line)
      - prev ends with a soft-hyphen
      - prev's last word is a conjunction/article/preposition/reference-word
        (fragment is syntactically incomplete)
      - prev does NOT end with sentence-terminating punctuation AND next
        starts with a digit (numeric continuation of a reference — e.g.
        "…and Chapter" + "19", or "2 Corinthians" + "12:11")
    """
    if not next_text:
        return False
    prev_text = prev_text.rstrip()
    first = next_text[0]
    if first.islower():
        return True
    if first in _CONTINUATION_LEADERS:
        return True
    if prev_text.count("(") > prev_text.count(")"):
        return True
    if prev_text.endswith(("-", "—", "–")):
        return True
    if _ends_with_sentence_terminator(prev_text):
        return False
    if _last_word(prev_text).lower() in _CONTINUATION_TAIL_WORDS:
        return True
    if first.isdigit():
        return True
    return False


def _looks_like_list_continuation(text: str, prev_item: ListItem) -> bool:
    prev_text = "".join(r.text for r in prev_item.runs)
    return _looks_like_wrap_continuation(prev_text, text)


def _append_text_to_item(item: ListItem, text: str, bold: bool, italic: bool) -> None:
    """Append `text` onto the item, merging into the last run when styles match."""
    if not text:
        return
    if item.runs and item.runs[-1].bold == bold and item.runs[-1].italic == italic:
        last = item.runs[-1]
        if last.text.endswith("-") and text and text[0].islower():
            last.text = last.text[:-1] + text
        else:
            sep = "" if last.text.endswith(("–", "—")) else _CONTINUATION_JOIN
            last.text = last.text + sep + text
    else:
        item.runs.append(TextRun(text=text, bold=bold, italic=italic))


def _extend_list_item(item: ListItem, cl: ClassifiedLine) -> None:
    """Append a wrapped-continuation line's whole text onto the item."""
    _append_text_to_item(
        item,
        cl.text.strip(),
        bold=cl.line.bold_ratio > 0.5,
        italic=cl.line.italic_ratio > 0.5,
    )


def _split_bullet_content(text: str) -> list[str]:
    """Split a bullet item's content on interior bullet delimiters.

    "Foo • Bar • Baz" → ["Foo", "Bar", "Baz"]
    "Foo •"           → ["Foo", ""]   (trailing empty ⇒ dangling bullet)
    "Foo"             → ["Foo"]
    """
    return [p.strip() for p in _INTERIOR_BULLET_RE.split(text)]


def _collect_list(
    lines: list[ClassifiedLine], start: int, style: str
) -> tuple[ListBlock | None, int]:
    """Collect a contiguous list starting at lines[start], allowing short
    interruptions (a callout or pull-quote) between items. Returns the block
    and how many source lines were consumed."""

    segments: list = []
    current_items: list[ListItem] = []
    interruption_buffer: list[Block] = []
    interruption_start_i: int | None = None
    starting_num = 1
    pending_new_item = False

    first_text = lines[start].text.strip()
    if style == "ordered":
        m = _ORDERED_ITEM_RE.match(first_text)
        if m and m.group(1).isdigit():
            starting_num = int(m.group(1))

    def _flush_interruption() -> None:
        nonlocal interruption_buffer, interruption_start_i
        if not interruption_buffer:
            return
        if current_items:
            segments.append(ItemSegment(items=list(current_items)))
            current_items.clear()
        segments.append(InterruptionSegment(blocks=list(interruption_buffer)))
        interruption_buffer = []
        interruption_start_i = None

    def _add_items_from_content(content: str) -> None:
        """Append one or more ListItems from `content`, splitting on interior
        bullets for unordered lists. Updates `pending_new_item` when the
        content ends with a dangling bullet."""
        nonlocal pending_new_item
        if style != "unordered":
            current_items.append(ListItem(runs=[_run(content)]))
            pending_new_item = False
            return
        segs = _split_bullet_content(content)
        pending_new_item = bool(segs) and segs[-1] == ""
        for s in segs:
            if s:
                current_items.append(ListItem(runs=[_run(s)]))

    i = start
    while i < len(lines):
        cl = lines[i]
        text = cl.text.strip()

        if cl.label in ("chapter-caption", "chapter-title", "part-marker", "part-title"):
            break

        item_match = _detect_list_item(text) if cl.label == "body" else None

        if item_match and item_match[0] == style:
            _flush_interruption()
            _add_items_from_content(item_match[1])
            i += 1
            continue

        if item_match:
            break

        # Previous item ended with a dangling `•` — this body line's text is
        # the start of a new item (whose bullet was consumed by the wrap on
        # the previous PDF line). Split on any interior bullets it also has.
        if (
            cl.label == "body"
            and current_items
            and not interruption_buffer
            and pending_new_item
        ):
            _add_items_from_content(text)
            i += 1
            continue

        # Wrapped-continuation: a body line whose text starts with lowercase
        # or a closing bracket/quote/punctuation (or where the previous item
        # ended mid-citation with an unmatched "(") is almost certainly the
        # tail of a wrapped list item, not the start of interrupting content.
        # Merge it into the previous ListItem instead of ending the list.
        # If the continuation also carries interior bullets (multiple items
        # packed into the same wrapped PDF line), split them off as new items.
        if (
            cl.label == "body"
            and current_items
            and not interruption_buffer
            and _looks_like_list_continuation(text, current_items[-1])
        ):
            if style == "unordered":
                segs = _split_bullet_content(text)
                bold = cl.line.bold_ratio > 0.5
                italic = cl.line.italic_ratio > 0.5
                if segs:
                    _append_text_to_item(current_items[-1], segs[0], bold, italic)
                    pending_new_item = segs[-1] == "" and len(segs) > 1
                    for s in segs[1:]:
                        if s:
                            current_items.append(ListItem(runs=[_run(s)]))
            else:
                _extend_list_item(current_items[-1], cl)
            i += 1
            continue

        if interruption_start_i is None:
            interruption_start_i = i

        if cl.label == "body":
            # Batch consecutive body lines into one Paragraph. Without this,
            # multi-line prose commentary that sits between list items (e.g. a
            # numbered list where each heading is followed by a 3-line
            # explanation) fragments into separate one-line <p> elements.
            body_lines = [cl]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.label != "body":
                    break
                if _detect_list_item(nxt.text.strip()):
                    break
                if _is_paragraph_break(body_lines[-1], nxt):
                    break
                body_lines.append(nxt)
                j += 1
            interruption_buffer.append(_build_paragraph(body_lines))
            i = j
        else:
            block = _line_to_simple_block(cl)
            if block is not None:
                interruption_buffer.append(block)
            i += 1

        if len(interruption_buffer) > 4:
            break

    if interruption_buffer:
        # Trailing non-list content — roll back, list ended before it.
        assert interruption_start_i is not None
        i = interruption_start_i
        interruption_buffer = []

    if current_items:
        segments.append(ItemSegment(items=list(current_items)))

    if not segments or not any(isinstance(s, ItemSegment) for s in segments):
        return None, 1

    return ListBlock(style=style, segments=segments, start=starting_num), i - start


# --------------------------------------------------------------------------- #
# Pull-quote merging + role assignment
# --------------------------------------------------------------------------- #

def _merge_pullquote_runs(prev: list[TextRun], nxt: list[TextRun]) -> list[TextRun]:
    combined = list(prev)
    for r in nxt:
        text = r.text
        if not text:
            continue
        if combined and combined[-1].bold == r.bold and combined[-1].italic == r.italic:
            last = combined[-1]
            if last.text.endswith("-") and text and text[0].islower():
                last.text = last.text[:-1] + text
            else:
                sep = "" if last.text.endswith(("–", "—")) else _CONTINUATION_JOIN
                last.text = last.text + sep + text
        else:
            # Different-style run following an existing run: still need an
            # inter-word space unless the previous run already ends in one or
            # the new text starts with punctuation.
            if combined:
                last = combined[-1]
                needs_space = (
                    last.text
                    and not last.text.endswith((" ", "-", "–", "—"))
                    and text[0] not in (" ", ",", ";", ":", ".", "?", "!", ")", "]", "}", "”", "’", "'", '"')
                )
                if needs_space:
                    text = " " + text
            combined.append(TextRun(text=text, bold=r.bold, italic=r.italic))
    return combined


def _merge_adjacent_pullquotes(blocks: list[Block]) -> list[Block]:
    """Merge line-wrap fragments back into their originating PullQuote.

    Handles two cases produced by PDF wrap:
      - two consecutive PullQuotes (the classifier tagged both as italic
        + centered).
      - a PullQuote followed by a Paragraph whose text is clearly the tail
        of the quote (the classifier lost the italic signal on the last
        line, so it got promoted to plain body).
    """
    out: list[Block] = []
    for b in blocks:
        if (
            isinstance(b, PullQuote)
            and out
            and isinstance(out[-1], PullQuote)
            and out[-1].attribution is None
            and b.attribution is None
        ):
            out[-1] = PullQuote(
                runs=_merge_pullquote_runs(out[-1].runs, b.runs),
            )
            continue
        if (
            isinstance(b, Paragraph)
            and out
            and isinstance(out[-1], PullQuote)
            and out[-1].attribution is None
        ):
            prev_text = _pullquote_text(out[-1])
            next_text = "".join(r.text for r in b.runs)
            if _looks_like_wrap_continuation(prev_text, next_text):
                # The paragraph is the tail of the quote whose leading lines
                # were italic; last-line italic often gets misdetected because
                # the PDF renderer switched font mid-line. Coerce italic — but
                # NOT onto bold runs, which are almost always the standalone
                # bold scripture citation ("— Matthew 9:37-38 (NKJV)") whose
                # correct rendering is <strong> alone, not <strong><em>.
                prev_italic = any(r.italic for r in out[-1].runs)
                coerced = [
                    TextRun(
                        text=r.text,
                        bold=r.bold,
                        italic=r.italic or (prev_italic and not r.bold),
                        small_caps=r.small_caps,
                        link=r.link,
                    )
                    for r in b.runs
                ]
                out[-1] = PullQuote(
                    runs=_merge_pullquote_runs(out[-1].runs, coerced),
                )
                continue
        out.append(b)
    return out


def _pullquote_text(q: PullQuote) -> str:
    return " ".join(r.text for r in q.runs)


def _all_italic(runs: list[TextRun]) -> bool:
    """True iff every non-empty run is italic (and none is bold)."""
    non_empty = [r for r in runs if r.text.strip()]
    if not non_empty:
        return False
    return all(r.italic and not r.bold for r in non_empty)


def _all_bold(runs: list[TextRun]) -> bool:
    non_empty = [r for r in runs if r.text.strip()]
    if not non_empty:
        return False
    return all(r.bold for r in non_empty)


def _is_scripture_citation_line(text: str) -> bool:
    """True iff `text` looks like a standalone citation ("— Psalm 33:9, NKJV")."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] not in "—–-":
        return False
    return _is_scripture_text(stripped)


def _strip_citation_dash(text: str) -> str:
    """Drop the leading em/en/hyphen dash + whitespace from a citation string."""
    return text.lstrip("—–- \t")


def _pair_scripture_citations(blocks: list[Block]) -> list[Block]:
    """Merge italic-verse + bold-citation pairs into `scripture` PullQuotes.

    Source manuscripts render each scripture as two visually-separate lines:
    an italic verse followed by a right-aligned bold "— Book chap:verse, VER"
    citation. The classifier gives us either (Paragraph italic) + (Paragraph
    bold) or (PullQuote plain) + (Paragraph bold). This pass fuses either
    shape into a single scripture PullQuote with the citation as its
    `attribution`, so the writer wraps it in <blockquote class="scripture">
    with the border/box styling defined in styles.css.
    """
    out: list[Block] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None

        verse_runs: list[TextRun] | None = None
        if isinstance(b, Paragraph) and _all_italic(b.runs):
            verse_runs = b.runs
        elif isinstance(b, PullQuote) and b.role == "plain" and _all_italic(b.runs):
            verse_runs = b.runs

        if (
            verse_runs is not None
            and isinstance(nxt, Paragraph)
            and _all_bold(nxt.runs)
        ):
            cite_text = "".join(r.text for r in nxt.runs).strip()
            if _is_scripture_citation_line(cite_text):
                attribution = _strip_citation_dash(cite_text)
                out.append(PullQuote(
                    runs=list(verse_runs),
                    attribution=attribution,
                    role="scripture",
                ))
                i += 2
                continue
        out.append(b)
        i += 1
    return out


def _is_scripture_text(text: str) -> bool:
    if _SCRIPTURE_TRANS_RE.search(text):
        return True
    if _SCRIPTURE_CITE_RE.search(text):
        return True
    return False


def _assign_pullquote_roles(blocks: list[Block]) -> None:
    """Reclassify PullQuote.role in place.

    Rules:
      - Any pull-quote whose text contains a scripture citation → scripture.
      - Leading pull-quote(s) at the top of the block list (immediately after
        the section title, before any body) → subtitle-tagline.
      - A pull-quote that comes right after the leading subtitle-tagline
        block(s) and an optional scene-break → epigraph.
      - Everything else → plain (centered italic, no border).
    """
    for b in blocks:
        if isinstance(b, PullQuote) and _is_scripture_text(_pullquote_text(b)):
            b.role = "scripture"

    lead_end = 0
    while (
        lead_end < len(blocks)
        and isinstance(blocks[lead_end], PullQuote)
        and blocks[lead_end].role == "plain"
    ):
        blocks[lead_end].role = "subtitle-tagline"
        lead_end += 1

    if lead_end == 0 or lead_end >= len(blocks):
        return

    j = lead_end
    if isinstance(blocks[j], SceneBreak):
        j += 1
    if (
        j < len(blocks)
        and isinstance(blocks[j], PullQuote)
        and blocks[j].role == "plain"
    ):
        blocks[j].role = "epigraph"


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build_book(items: list[ClassifiedLine], metadata: BookMetadata) -> Book:
    front, parts, loose_chapters, back = _segment_sections(items)

    slug_inputs: list[str] = []
    for sec in front:
        slug_inputs.append(f"front-{slugify(sec.title)}")
    for pidx, prec in enumerate(parts, start=1):
        slug_inputs.append(f"part-{pidx:02d}-{slugify(prec.title)}")
        for cidx, chap in enumerate(prec.chapters, start=1):
            slug_inputs.append(f"ch-{pidx:02d}-{cidx:02d}-{slugify(chap.title)}")
    for cidx, chap in enumerate(loose_chapters, start=1):
        slug_inputs.append(f"ch-{cidx:02d}-{slugify(chap.title)}")
    for sec in back:
        slug_inputs.append(f"back-{slugify(sec.title)}")
    unique_slugs = iter(_make_slugs_unique(slug_inputs))

    front_sections: list[Section] = []
    for sec in front:
        built = _build_section(sec, next(unique_slugs))
        if _is_meaningful_section(built):
            front_sections.append(built)

    part_objs: list[Part] = []
    demoted_back: list[Section] = []
    for prec in parts:
        part_slug = next(unique_slugs)
        chapter_sections: list[Section] = []
        for chap in prec.chapters:
            built = _build_section(chap, next(unique_slugs))
            if _is_meaningful_section(built):
                chapter_sections.append(built)
        intro_blocks = _convert_body_to_blocks(prec.intro_lines)
        intro_blocks = _merge_adjacent_pullquotes(intro_blocks)
        intro_blocks = _pair_scripture_citations(intro_blocks)
        _assign_pullquote_roles(intro_blocks)
        if chapter_sections:
            part_objs.append(Part(
                marker=prec.marker,
                title=prec.title,
                chapters=chapter_sections,
                slug=part_slug,
                intro=intro_blocks,
            ))
        elif _content_has_substance(intro_blocks):
            # Part-title with no chapter subdivisions (common when the
            # classifier tags e.g. "APPENDICES" or "FINAL WORD" as
            # part-title but the content underneath is a single flat
            # section). Demote to a back-matter section so nothing is lost.
            demoted_back.append(Section(
                title=prec.title,
                kind="back",
                slug=part_slug,
                content=intro_blocks,
            ))

    loose_chapter_sections: list[Section] = []
    for chap in loose_chapters:
        built = _build_section(chap, next(unique_slugs))
        if _is_meaningful_section(built):
            loose_chapter_sections.append(built)

    back_sections: list[Section] = list(demoted_back)
    for sec in back:
        built = _build_section(sec, next(unique_slugs))
        if _is_meaningful_section(built):
            back_sections.append(built)

    # `has_parts` gating in _segment_sections guarantees loose_chapters is
    # only populated when there are no part_objs, so Book's parts/chapters
    # invariant holds without further reshuffling.
    assert not (part_objs and loose_chapter_sections), (
        "internal error: loose chapters coexist with parts"
    )

    return Book(
        metadata=metadata,
        front_matter=front_sections,
        parts=part_objs,
        chapters=loose_chapter_sections,
        back_matter=back_sections,
    )


def _build_section(flat: _FlatSection, slug: str) -> Section:
    join = not _looks_like_toc_title(flat.title)
    content = _convert_body_to_blocks(flat.body_lines, join_paragraphs=join)
    content = _merge_adjacent_pullquotes(content)
    content = _pair_scripture_citations(content)
    _assign_pullquote_roles(content)
    return Section(
        title=flat.title,
        kind=flat.kind,  # type: ignore[arg-type]
        slug=slug,
        content=content,
        caption=flat.caption,
    )
