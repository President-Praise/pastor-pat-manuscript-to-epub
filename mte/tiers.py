"""Feature-based tier detection.

Replaces raw-size ordering with per-size feature vectors and role scoring.
Roles: body, cover-title, part-title, chapter, h2, h3, pull-quote-tier,
sub-body.

Body is always the char-weight mode. Everything else is assigned by looking
at how each size is USED — how many pages it appears on, centered ratio,
bold/caps ratios, average line length. That way a size used only twice for
a cover ornament cannot steal the 'chapter' slot from the size that is
actually used for every chapter title.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .pdf_extract import Line


@dataclass
class SizeFeatures:
    size: float
    line_count: int = 0
    char_count: int = 0
    distinct_pages: int = 0
    centered_count: int = 0
    bold_count: int = 0
    italic_count: int = 0
    caps_count: int = 0
    total_text_len: int = 0

    @property
    def centered_ratio(self) -> float:
        return self.centered_count / max(1, self.line_count)

    @property
    def bold_ratio(self) -> float:
        return self.bold_count / max(1, self.line_count)

    @property
    def italic_ratio(self) -> float:
        return self.italic_count / max(1, self.line_count)

    @property
    def caps_ratio(self) -> float:
        return self.caps_count / max(1, self.line_count)

    @property
    def avg_text_len(self) -> float:
        return self.total_text_len / max(1, self.line_count)


@dataclass
class TierAssignment:
    body: float
    size_to_role: dict[float, str] = field(default_factory=dict)
    features: dict[float, SizeFeatures] = field(default_factory=dict)
    part_size: float | None = None
    chapter_size: float | None = None
    h2_size: float | None = None
    h3_size: float | None = None
    cover_sizes: set[float] = field(default_factory=set)

    def role_for(self, size: float, tolerance: float = 0.4) -> str:
        best = None
        best_d = 1e9
        for s in self.size_to_role:
            d = abs(size - s)
            if d < best_d:
                best_d = d
                best = s
        if best is None or best_d > tolerance:
            if abs(size - self.body) <= tolerance:
                return "body"
            return "unknown-large" if size > self.body else "sub-body"
        return self.size_to_role[best]


def _is_mostly_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) > 0.85


_PART_RE = re.compile(r"^\s*PART\s+([IVXLCM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b", re.IGNORECASE)
_CHAP_CAP_RE = re.compile(r"^\s*(CHAPTER\s+\d+|CH\.?\s*\d+|C\d+|\d+)\s*$", re.IGNORECASE)


def _skip_from_features(ln: Line, body: float | None) -> bool:
    text = ln.text.strip()
    if not text:
        return True
    if _PART_RE.match(text):
        return True
    if _CHAP_CAP_RE.match(text):
        return True
    if body is not None and abs(ln.dominant_size - body) <= 2.5:
        if ln.italic_ratio > 0.5 and ln.is_centered:
            return True
    return False


def compute_features(lines: Iterable[Line], body: float | None = None) -> dict[float, SizeFeatures]:
    features: dict[float, SizeFeatures] = {}
    pages_by_size: dict[float, set[int]] = {}
    for ln in lines:
        text = ln.text.strip()
        if not text:
            continue
        if _skip_from_features(ln, body):
            continue
        size = ln.dominant_size
        f = features.setdefault(size, SizeFeatures(size=size))
        f.line_count += 1
        f.char_count += len(text)
        f.total_text_len += len(text)
        pages_by_size.setdefault(size, set()).add(ln.page)
        if ln.is_centered:
            f.centered_count += 1
        if ln.bold_ratio > 0.5:
            f.bold_count += 1
        if ln.italic_ratio > 0.5:
            f.italic_count += 1
        if _is_mostly_caps(text):
            f.caps_count += 1
    for size, f in features.items():
        f.distinct_pages = len(pages_by_size[size])
    return features


def _pick_body(features: dict[float, SizeFeatures]) -> float:
    return max(features, key=lambda s: features[s].char_count)


def assign_tiers(lines: list[Line]) -> TierAssignment:
    pass_a = compute_features(lines, body=None)
    if not pass_a:
        return TierAssignment(body=0.0)
    body = _pick_body(pass_a)
    features = compute_features(lines, body=body)
    if body not in features:
        features[body] = pass_a[body]

    above = [f for s, f in features.items() if s > body + 0.4]
    above.sort(key=lambda f: -f.size)

    cover_sizes: set[float] = set()
    heading_candidates: list[SizeFeatures] = []
    for f in above:
        if f.distinct_pages <= 2 and f.line_count <= 3:
            cover_sizes.add(f.size)
        else:
            heading_candidates.append(f)

    def is_chapterish(f: SizeFeatures) -> bool:
        return (
            f.centered_ratio >= 0.7
            and (f.bold_ratio >= 0.7 or f.caps_ratio >= 0.7)
            and f.avg_text_len <= 90
        )

    def is_pull_quote_ish(f: SizeFeatures) -> bool:
        return f.italic_ratio >= 0.5 and f.centered_ratio >= 0.5 and f.size <= body + 3.5

    def score_chapter(f: SizeFeatures) -> float:
        s = 0.0
        s += min(f.distinct_pages, 40) * 0.25
        s += f.centered_ratio * 2.0
        s += f.bold_ratio * 1.5
        s += f.caps_ratio * 1.0
        if 4 <= f.avg_text_len <= 60:
            s += 2.0
        elif f.avg_text_len > 120:
            s -= 3.0
        s += (f.size - body) * 0.5
        return s

    valid_chapters = [f for f in heading_candidates if is_chapterish(f) and f.distinct_pages >= 3]
    chapter_size = None
    if valid_chapters:
        chapter_size = max(valid_chapters, key=score_chapter).size

    part_size = None
    if chapter_size is not None:
        above_chapter = [
            f for f in heading_candidates
            if f.size > chapter_size + 0.4 and is_chapterish(f)
        ]
        if above_chapter:
            part_size = max(above_chapter, key=lambda f: f.size).size

    h2_size = None
    h3_size = None
    below_chapter = sorted(
        (
            f for f in heading_candidates
            if (chapter_size is None or f.size < chapter_size - 0.4)
            and f.bold_ratio >= 0.5
            and not is_pull_quote_ish(f)
        ),
        key=lambda f: -f.size,
    )
    if below_chapter:
        h2_size = below_chapter[0].size
        if len(below_chapter) > 1:
            h3_size = below_chapter[1].size

    pull_quote_sizes = {f.size for f in heading_candidates if is_pull_quote_ish(f)}

    size_to_role: dict[float, str] = {body: "body"}
    for s in cover_sizes:
        size_to_role[s] = "cover-title"
    if part_size is not None:
        size_to_role[part_size] = "part-title"
    if chapter_size is not None:
        size_to_role[chapter_size] = "chapter"
    if h2_size is not None:
        size_to_role[h2_size] = "h2"
    if h3_size is not None:
        size_to_role[h3_size] = "h3"
    for s in pull_quote_sizes:
        size_to_role.setdefault(s, "pull-quote-tier")
    for s in features:
        if s not in size_to_role and s > body + 0.4:
            size_to_role[s] = "unknown-large"
        elif s not in size_to_role and s < body - 0.4:
            size_to_role[s] = "sub-body"

    return TierAssignment(
        body=body,
        size_to_role=size_to_role,
        features=features,
        part_size=part_size,
        chapter_size=chapter_size,
        h2_size=h2_size,
        h3_size=h3_size,
        cover_sizes=cover_sizes,
    )


def format_tier_report(assign: TierAssignment) -> str:
    out = ["Feature-based tier assignment:"]
    out.append(f"  body size: {assign.body:.1f} pt")
    out.append(f"  cover-title sizes: {sorted(assign.cover_sizes, reverse=True)}")
    out.append(f"  part-title:  {assign.part_size}")
    out.append(f"  chapter:     {assign.chapter_size}")
    out.append(f"  h2:          {assign.h2_size}")
    out.append(f"  h3:          {assign.h3_size}")
    out.append("")
    out.append("Per-size features (sorted by size desc):")
    out.append(f"  {'size':>5}  {'role':>15}  {'lines':>6}  {'pages':>5}  {'ctr':>5}  {'bold':>5}  {'caps':>5}  {'it':>5}  {'avglen':>6}")
    for size in sorted(assign.features, reverse=True):
        f = assign.features[size]
        role = assign.role_for(size)
        out.append(
            f"  {size:>5.1f}  {role:>15}  {f.line_count:>6d}  {f.distinct_pages:>5d}  "
            f"{f.centered_ratio:>5.2f}  {f.bold_ratio:>5.2f}  {f.caps_ratio:>5.2f}  "
            f"{f.italic_ratio:>5.2f}  {f.avg_text_len:>6.1f}"
        )
    return "\n".join(out)
