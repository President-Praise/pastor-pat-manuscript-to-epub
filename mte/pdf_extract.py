"""Extract typeset spans from a PDF using PyMuPDF.

Ground-truth extraction: every text run keeps its font name, size, bold/italic
flags, bbox, page number, and page width. We do NOT flatten to plain text —
downstream classification is font-metric-driven.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterator

import fitz  # PyMuPDF


# PyMuPDF span flag bits (see fitz docs). We only care about a subset.
FLAG_SUPERSCRIPT = 1 << 0
FLAG_ITALIC      = 1 << 1
FLAG_SERIFED     = 1 << 2
FLAG_MONO        = 1 << 3
FLAG_BOLD        = 1 << 4


@dataclass
class Span:
    text: str
    font: str
    size: float          # rounded to 0.1 pt
    bold: bool
    italic: bool
    x0: float
    x1: float
    y0: float
    y1: float
    page: int            # 1-based
    page_width: float
    page_height: float
    char_xs: list[tuple[float, float]] | None = None  # per-char (x0, x1); populated only when text has internal spaces

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class Line:
    """A visual line, assembled from consecutive spans on the same baseline."""
    spans: list[Span] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)

    @property
    def page(self) -> int:
        return self.spans[0].page

    @property
    def page_width(self) -> float:
        return self.spans[0].page_width

    @property
    def page_height(self) -> float:
        return self.spans[0].page_height

    @property
    def dominant_size(self) -> float:
        # Weight by character count so a stray tiny footnote marker doesn't
        # dominate a normal-sized line.
        if not self.spans:
            return 0.0
        totals: dict[float, int] = {}
        for s in self.spans:
            totals[s.size] = totals.get(s.size, 0) + max(1, len(s.text))
        return max(totals, key=lambda k: totals[k])

    @property
    def bold_ratio(self) -> float:
        chars = sum(len(s.text) for s in self.spans) or 1
        return sum(len(s.text) for s in self.spans if s.bold) / chars

    @property
    def italic_ratio(self) -> float:
        chars = sum(len(s.text) for s in self.spans) or 1
        return sum(len(s.text) for s in self.spans if s.italic) / chars

    @property
    def x0(self) -> float:
        return min(s.x0 for s in self.spans)

    @property
    def x1(self) -> float:
        return max(s.x1 for s in self.spans)

    @property
    def y0(self) -> float:
        return min(s.y0 for s in self.spans)

    @property
    def y1(self) -> float:
        return max(s.y1 for s in self.spans)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def is_centered(self) -> bool:
        # Centered if the line's center is within a small band around the
        # page center AND there's meaningful whitespace on both sides.
        page_center = self.page_width / 2.0
        # Allow ~5% of page width tolerance
        tol = self.page_width * 0.05
        if abs(self.center_x - page_center) > tol:
            return False
        left_margin = self.x0
        right_margin = self.page_width - self.x1
        # Both margins should be similarly-sized and non-trivial
        if min(left_margin, right_margin) < self.page_width * 0.05:
            return False
        return abs(left_margin - right_margin) < self.page_width * 0.03


def _is_bold_font(font_name: str, flags: int) -> bool:
    if flags & FLAG_BOLD:
        return True
    lower = font_name.lower()
    return (
        "bold" in lower
        or "black" in lower
        or "heavy" in lower
        or "semibold" in lower
        or lower.endswith("-bd")
    )


def _is_italic_font(font_name: str, flags: int) -> bool:
    if flags & FLAG_ITALIC:
        return True
    lower = font_name.lower()
    return "italic" in lower or "oblique" in lower


def extract_spans(pdf_path: str) -> list[Span]:
    """Read every text span from the PDF, preserving typesetting attributes."""
    spans: list[Span] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            page_width = page.rect.width
            page_height = page.rect.height
            page_no = page_index + 1
            raw = page.get_text("rawdict")
            for block in raw.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    for s in line.get("spans", []):
                        chars = s.get("chars", [])
                        text = "".join(c.get("c", "") for c in chars)
                        if not text:
                            continue
                        flags = int(s.get("flags", 0))
                        font_name = s.get("font", "") or ""
                        size = round(float(s.get("size", 0.0)), 1)
                        x0, y0, x1, y1 = s.get("bbox", (0, 0, 0, 0))
                        char_xs: list[tuple[float, float]] | None = None
                        if len(chars) > 1:
                            char_xs = [
                                (float(c["bbox"][0]), float(c["bbox"][2]))
                                for c in chars
                            ]
                        spans.append(Span(
                            text=text,
                            font=font_name,
                            size=size,
                            bold=_is_bold_font(font_name, flags),
                            italic=_is_italic_font(font_name, flags),
                            x0=float(x0), x1=float(x1),
                            y0=float(y0), y1=float(y1),
                            page=page_no,
                            page_width=page_width,
                            page_height=page_height,
                            char_xs=char_xs,
                        ))
    return spans


def group_into_lines(spans: list[Span], y_tol: float = 2.0) -> list[Line]:
    """Group spans that share (page, baseline) into visual lines.

    We do it per-page and cluster by y0 rounded to `y_tol` points, then sort
    each cluster left-to-right by x0.
    """
    by_page: dict[int, list[Span]] = {}
    for s in spans:
        by_page.setdefault(s.page, []).append(s)

    lines: list[Line] = []
    for page in sorted(by_page):
        page_spans = by_page[page]
        # Cluster by baseline (y0). Bin to nearest y_tol.
        buckets: dict[int, list[Span]] = {}
        for s in page_spans:
            key = int(round(s.y0 / y_tol))
            buckets.setdefault(key, []).append(s)
        # Emit in top-to-bottom order.
        for key in sorted(buckets):
            group = sorted(buckets[key], key=lambda s: s.x0)
            lines.append(Line(spans=group))
    return lines


def span_to_dict(s: Span) -> dict:
    return asdict(s)


def iter_lines(pdf_path: str) -> Iterator[Line]:
    return iter(group_into_lines(extract_spans(pdf_path)))
