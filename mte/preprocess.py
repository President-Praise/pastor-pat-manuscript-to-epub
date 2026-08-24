"""Preprocess raw lines before classification.

Three passes, each reversible for reporting purposes:
    1. Repeated running headers/footers → dropped.
    2. Bare page-number lines near top/bottom margins → dropped.
    3. Drop-cap first-letter spans → merged back into their body line.
    4. Letter-spaced-caps text → normalized (span text rewritten in place).

We return the filtered line list PLUS a StripReport so the dry-run can show
what was removed. Silent stripping is the enemy of trust.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .pdf_extract import Line, Span


DECORATIVE_CHARS = set("✦❖❦♚♛✧✩✪★☆•●◆◇◈❧⚜❈❃❊§¶—–_")


def _is_wrap_continuation_of_prev(lines: list[Line], i: int) -> bool:
    """True if `lines[i]` is a same-size continuation of `lines[i-1]`.

    Drop-cap merges (single- and multi-letter) must not fire on a line that
    is really the *second visual line of a wrapped heading*. A two-line
    title like:
        line[i-1]: "BELIEVERS AS PRIESTS AND"   (28pt, bold)
        line[i]:   "KINGS"                       (28pt, bold)
        line[i+1]: "Recovering the Royal..."     (13pt, subtitle)
    exactly matches the drop-cap heuristic if we look only at (i, i+1) —
    "KINGS" is an all-caps short line above a smaller line. But it is not
    a drop-cap row; it is a wrap of the heading above it, and treating it
    otherwise scrambles the subtitle by prepending K/I/N/G/S letters into
    unrelated words. Guard: if the immediately-preceding kept line is
    same-page, same-size, same-bold, and vertically adjacent, this line
    belongs to that heading — bail out.
    """
    if i == 0:
        return False
    prev = lines[i - 1]
    ln = lines[i]
    if prev.page != ln.page:
        return False
    if abs(prev.dominant_size - ln.dominant_size) > 0.5:
        return False
    if (prev.bold_ratio > 0.5) != (ln.bold_ratio > 0.5):
        return False
    line_h = max(prev.dominant_size, ln.dominant_size)
    if (ln.y0 - prev.y1) > line_h * 1.5:
        return False
    return True


@dataclass
class StripReport:
    header_lines_removed: int = 0
    footer_lines_removed: int = 0
    page_number_lines_removed: int = 0
    drop_caps_merged: int = 0
    letter_spaced_normalized: int = 0
    header_examples: list[str] = field(default_factory=list)
    footer_examples: list[str] = field(default_factory=list)


def _norm_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip())


def _line_y_frac(line: Line) -> float:
    return line.y0 / max(1.0, line.page_height)


def _looks_like_page_number(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if re.fullmatch(r"\d{1,4}", t):
        return True
    if re.fullmatch(r"[—–\-]\s*\d{1,4}\s*[—–\-]", t):
        return True
    if len(t) >= 2 and re.fullmatch(r"[ivxlcIVXLC]{2,6}", t):
        return True
    return False


def detect_repeats(
    lines: list[Line], min_pages_ratio: float = 0.30, margin_frac: float = 0.12
) -> set[tuple[str, str]]:
    """Return the set of (norm_text, position) keys that repeat across pages.

    position ∈ {"TOP","BOT"}; a line is "TOP" if in the upper `margin_frac` of
    the page, "BOT" if in the lower `margin_frac`. Middle-of-page repetition
    is NOT considered (real content can repeat).
    """
    pages = {ln.page for ln in lines}
    if not pages:
        return set()
    threshold = max(3, int(len(pages) * min_pages_ratio))

    seen_on_pages: dict[tuple[str, str], set[int]] = defaultdict(set)
    for ln in lines:
        text = ln.text.strip()
        if not text or len(text) > 100:
            continue
        y_frac = _line_y_frac(ln)
        if y_frac < margin_frac:
            pos = "TOP"
        elif y_frac > 1.0 - margin_frac:
            pos = "BOT"
        else:
            continue
        seen_on_pages[(_norm_key(text), pos)].add(ln.page)

    return {k for k, pgs in seen_on_pages.items() if len(pgs) >= threshold}


def _looks_letter_spaced(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 5:
        return False
    tokens = stripped.split()
    if len(tokens) < 3:
        return False
    single_char = sum(1 for t in tokens if len(t) == 1 and (t.isalpha() or t.isdigit()))
    return single_char / len(tokens) >= 0.6


def _letter_positions_across_spans(line: Line) -> list[tuple[str, float, float]]:
    """Return (char, x0, x1) triples for every alphanumeric glyph on the line.

    Sources positions from span.char_xs when available; falls back to
    even-width distribution across the span's bbox otherwise. Whitespace
    glyphs are skipped — word boundaries are recovered downstream from the
    x-gaps between consecutive letters, not from stored space characters.
    """
    def _is_letter(c: str) -> bool:
        # Anything that isn't whitespace counts as a glyph belonging to a
        # word — hyphens, apostrophes, ampersands, etc. must be preserved.
        return bool(c) and not c.isspace()

    out: list[tuple[str, float, float]] = []
    for span in line.spans:
        text = span.text
        if not text:
            continue
        if span.char_xs and len(span.char_xs) == len(text):
            for ch, (cx0, cx1) in zip(text, span.char_xs):
                if _is_letter(ch):
                    out.append((ch, cx0, cx1))
            continue
        # No per-char positions: approximate.
        letter_chars = [c for c in text if _is_letter(c)]
        if not letter_chars:
            continue
        width = span.x1 - span.x0
        if len(letter_chars) == len(text.strip()):
            step = width / max(1, len(letter_chars))
            for i, ch in enumerate(letter_chars):
                cx0 = span.x0 + i * step
                out.append((ch, cx0, cx0 + step))
        else:
            # Mixed spaces+letters without char_xs — best-effort: distribute
            # positions across the full span width by original character index.
            n = len(text)
            step = width / n
            for i, ch in enumerate(text):
                if _is_letter(ch):
                    cx0 = span.x0 + i * step
                    out.append((ch, cx0, cx0 + step))
    return out


def _normalize_letter_spaced_from_spans(line: Line) -> tuple[str, bool]:
    """Rebuild a letter-spaced-caps line into normal words.

    Extracts per-letter x-positions across the whole line (works whether
    letters come as one big span with char_xs or as many single-glyph
    spans), then uses gap analysis to distinguish intra-word letter
    spacing from inter-word gaps.
    """
    joined = "".join(s.text for s in line.spans)
    if not _looks_letter_spaced(joined):
        return joined, False

    letters = _letter_positions_across_spans(line)
    if len(letters) < 3:
        return joined, False

    gaps = [b[1] - a[2] for a, b in zip(letters, letters[1:])]
    median = sorted(gaps)[len(gaps) // 2]
    threshold = max(median + 2.0, median * 1.8, 2.0)

    rebuilt: list[str] = [letters[0][0]]
    for i, gap in enumerate(gaps):
        if gap >= threshold:
            rebuilt.append(" ")
        rebuilt.append(letters[i + 1][0])
    normalized = "".join(rebuilt).strip()
    if not normalized or normalized == joined.strip():
        return joined, False
    return normalized, True


def _merge_drop_caps(lines: list[Line]) -> tuple[list[Line], int]:
    """Merge a single-letter big-font span at the start of a line with the
    smaller-font tail spans that immediately follow it (same page, nearby y).

    A drop cap span is:
      - single alphabetic character (after strip)
      - bold-ish (font-size distinctly larger than the median size on that page)
      - its right edge is near or before the left edge of the next span
    Neighbor spans on the same visual paragraph then attach.

    Because PyMuPDF returns drop-caps on their own y0 (top-aligned to the tall
    letter), the line-grouper puts them in their own line. We fuse: the tail
    line absorbs the drop-cap character prepended.
    """
    if not lines:
        return lines, 0

    merged: list[Line] = []
    merges = 0
    skip_next = False
    for i, ln in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        text = ln.text.strip()
        if (
            len(text) == 1
            and text.isalpha()
            and i + 1 < len(lines)
            and not _is_wrap_continuation_of_prev(lines, i)
        ):
            nxt = lines[i + 1]
            if nxt.page == ln.page and 0 <= (nxt.y0 - ln.y0) < ln.page_height * 0.05:
                if ln.dominant_size > nxt.dominant_size * 1.2:
                    if nxt.x0 < ln.x1 + (nxt.page_width * 0.05):
                        new_span = Span(
                            text=text + nxt.text.lstrip(),
                            font=nxt.spans[0].font,
                            size=nxt.dominant_size,
                            bold=nxt.spans[0].bold,
                            italic=nxt.spans[0].italic,
                            x0=ln.x0,
                            x1=nxt.x1,
                            y0=nxt.y0,
                            y1=nxt.y1,
                            page=nxt.page,
                            page_width=nxt.page_width,
                            page_height=nxt.page_height,
                        )
                        rebuilt = Line(spans=[new_span] + nxt.spans[1:])
                        merged.append(rebuilt)
                        merges += 1
                        skip_next = True
                        continue
        merged.append(ln)
    return merged, merges


def _word_x_starts(line: Line) -> list[tuple[int, float, str]]:
    """Return (start_index_in_text, x-position, word) for each whitespace-separated
    word in a line, using per-span char_xs when available so positions are exact."""
    words: list[tuple[int, float, str]] = []
    text = line.text
    if not text.strip():
        return words

    char_positions: list[float | None] = [None] * len(text)
    cursor = 0
    for span in line.spans:
        n = len(span.text)
        if span.char_xs and len(span.char_xs) == n:
            for k in range(n):
                if cursor + k < len(text):
                    char_positions[cursor + k] = span.char_xs[k][0]
        else:
            if n > 0:
                width_per_char = (span.x1 - span.x0) / n
                for k in range(n):
                    if cursor + k < len(text):
                        char_positions[cursor + k] = span.x0 + k * width_per_char
        cursor += n

    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        start = i
        while i < len(text) and not text[i].isspace():
            i += 1
        x = char_positions[start] if char_positions[start] is not None else line.x0
        words.append((start, x, text[start:i]))
    return words


def _tail_is_all_caps_heading(text: str) -> bool:
    """True iff the tail candidate is an all-uppercase heading skeleton.

    Real drop-cap targets like 'OW TO SE HIS ANUAL' are entirely uppercase —
    the heading is 'HOW TO USE THIS MANUAL' with the first letter of each
    word replaced by a decorative drop cap on the line above. A tail that
    contains any lowercase letters (e.g. 'When the Sea is Raging, the
    Saints Must Not Be Sleeping', 'Recovering the Royal and Priestly
    Identity of the Church') is a normal sentence and must not have caps
    prepended to its words. Digits are allowed (headings can contain
    "PSALM 23") — only lowercase letters disqualify.
    """
    for c in text:
        if c.isalpha() and c.islower():
            return False
    return True


def _merge_split_initial_caps(lines: list[Line]) -> tuple[list[Line], int]:
    """Handle multi-letter initial-cap drop-cap layouts.

    Pattern: a line consisting of N uppercase letters (no spaces or a few tight
    spaces) sits directly above a heading line. Each drop-cap letter aligns
    horizontally with a word start in the heading. We prepend each aligned
    letter to its corresponding word to reconstruct the true heading text.
    Example: line1='HUTM' (18pt) + line2='OW TO SE HIS ANUAL' (14.5pt) →
    'HOW TO USE THIS MANUAL'.
    """
    if not lines:
        return lines, 0

    out: list[Line] = []
    merges = 0
    skip_next = False
    for i, ln in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        text = ln.text.strip()
        letters = [c for c in text if c.isalpha()]
        # Limit raised from 8 → 10 to cover the AUDACIOUS Ch 4-12 heading
        # "PRAYER FOR ASSIGNMENTS AND NATIONS: THE ELIJAH-JOSHUA-DANIEL-ESTHER
        # PATTERN" (9 drop caps: PANTEJDEP). NOTE: even at 10, the ELIJAH-JOSHUA
        # -DANIEL-ESTHER hyphens are still dropped by this merger, gluing the
        # tail words together (→ "ELIJAHJOSHUADANIELESTHER"). See TODO below.
        if (
            2 <= len(letters) <= 10
            and all(c.isupper() for c in letters)
            and not any(c.isdigit() for c in text)
            and i + 1 < len(lines)
            and not _is_wrap_continuation_of_prev(lines, i)
        ):
            nxt = lines[i + 1]
            if (
                nxt.page == ln.page
                and 0 <= (nxt.y0 - ln.y0) < ln.page_height * 0.05
                and ln.dominant_size > nxt.dominant_size * 1.15
                and _tail_is_all_caps_heading(nxt.text)
            ):
                cap_positions: list[tuple[float, str]] = []
                if len(ln.spans) == 1:
                    cap_span = ln.spans[0]
                    if cap_span.char_xs and len(cap_span.char_xs) == len(cap_span.text):
                        cap_positions = [
                            (cap_span.char_xs[k][0], c)
                            for k, c in enumerate(cap_span.text)
                            if c.isalpha()
                        ]
                    else:
                        width = cap_span.x1 - cap_span.x0
                        step = width / max(1, len(letters))
                        cap_positions = [
                            (cap_span.x0 + step * (k + 0.5), c)
                            for k, c in enumerate(letters)
                        ]
                else:
                    for s in ln.spans:
                        st = s.text.strip()
                        if len(st) == 1 and st.isalpha() and st.isupper():
                            cap_positions.append((s.x0, st))

                if cap_positions:
                    tail_words = _word_x_starts(nxt)
                    if tail_words and cap_positions:
                        used = set()
                        assignments: dict[int, str] = {}
                        for cap_x, cap_ch in cap_positions:
                            best_j = None
                            best_d = 1e9
                            for j, (_, wx, _) in enumerate(tail_words):
                                if j in used:
                                    continue
                                d = abs(wx - cap_x)
                                if d < best_d:
                                    best_d = d
                                    best_j = j
                            if best_j is not None and best_d < ln.page_width * 0.1:
                                used.add(best_j)
                                assignments[best_j] = cap_ch

                        if assignments:
                            rebuilt_words = []
                            for j, (_, _, w) in enumerate(tail_words):
                                if j in assignments:
                                    rebuilt_words.append(assignments[j] + w)
                                else:
                                    rebuilt_words.append(w)
                            rebuilt_text = " ".join(rebuilt_words)
                            new_span = Span(
                                text=rebuilt_text,
                                font=nxt.spans[0].font,
                                size=nxt.dominant_size,
                                bold=any(s.bold for s in nxt.spans),
                                italic=any(s.italic for s in nxt.spans),
                                x0=ln.x0,
                                x1=nxt.x1,
                                y0=nxt.y0,
                                y1=nxt.y1,
                                page=nxt.page,
                                page_width=nxt.page_width,
                                page_height=nxt.page_height,
                            )
                            out.append(Line(spans=[new_span]))
                            merges += 1
                            skip_next = True
                            continue
        out.append(ln)
    return out, merges

# TODO(drop-cap hyphens): The tail line's small-caps text uses hyphens as
# word separators in some AUDACIOUS Ch 4-12 headings, e.g. the source
#   "PRAYER FOR ASSIGNMENTS AND NATIONS: THE ELIJAH-JOSHUA-DANIEL-ESTHER
#    PATTERN"
# gets extracted as one visual line where "ELIJAH-JOSHUA-DANIEL-ESTHER" is
# a single hyphen-joined token. _word_x_starts splits on whitespace only, so
# it sees ONE word starting at the E position. The first drop cap E lands,
# but the caps for J, D, and the second E have nowhere to go and are
# dropped, and the internal hyphens are also lost — giving the observed
# post-fix output
#   "PRAYER FOR ASSIGNMENTS AND NATIONS THE ELIJAHOSHUAANIELSTHER PATTERN"
# (three missing caps + three missing hyphens; four names glued into one).
# Fix: split _word_x_starts on hyphens as well as whitespace, then re-emit
# hyphens between the assigned caps.
#
# Same fix will restore the missing hyphen in
#   "PRAYER FOR CHILDREN AND FAMILIES: THE HANNAH-SYROPHOENICIAN PATTERN"
# currently emitted as "…THE HANNAHYROPHOENICIAN PATTERN".


def preprocess(lines: list[Line]) -> tuple[list[Line], StripReport]:
    report = StripReport()

    repeat_keys = detect_repeats(lines)

    kept: list[Line] = []
    for ln in lines:
        text = ln.text.strip()
        if not text:
            continue
        y_frac = _line_y_frac(ln)
        pos = "TOP" if y_frac < 0.12 else ("BOT" if y_frac > 0.88 else None)

        if pos is not None and (_norm_key(text), pos) in repeat_keys:
            if pos == "TOP":
                report.header_lines_removed += 1
                if len(report.header_examples) < 3:
                    report.header_examples.append(text[:80])
            else:
                report.footer_lines_removed += 1
                if len(report.footer_examples) < 3:
                    report.footer_examples.append(text[:80])
            continue

        if pos is not None and _looks_like_page_number(text):
            report.page_number_lines_removed += 1
            continue

        kept.append(ln)

    kept, merged = _merge_drop_caps(kept)
    report.drop_caps_merged = merged

    kept, initial_cap_merges = _merge_split_initial_caps(kept)
    report.drop_caps_merged += initial_cap_merges

    for ln in kept:
        combined = ln.text
        normalized, changed = _normalize_letter_spaced_from_spans(ln)
        if changed and normalized != combined:
            ln.spans[0] = Span(
                text=normalized,
                font=ln.spans[0].font,
                size=ln.dominant_size,
                bold=any(s.bold for s in ln.spans),
                italic=any(s.italic for s in ln.spans),
                x0=ln.x0,
                x1=ln.x1,
                y0=ln.y0,
                y1=ln.y1,
                page=ln.page,
                page_width=ln.page_width,
                page_height=ln.page_height,
            )
            del ln.spans[1:]
            report.letter_spaced_normalized += 1

    return kept, report


def is_decorative_only(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return all(c in DECORATIVE_CHARS or c.isspace() for c in t)


def format_strip_report(report: StripReport) -> str:
    lines = ["Preprocess strip report:"]
    lines.append(f"  running headers dropped: {report.header_lines_removed}")
    for ex in report.header_examples:
        lines.append(f"    ex: {ex}")
    lines.append(f"  running footers dropped: {report.footer_lines_removed}")
    for ex in report.footer_examples:
        lines.append(f"    ex: {ex}")
    lines.append(f"  bare page-number lines dropped: {report.page_number_lines_removed}")
    lines.append(f"  drop-caps merged into next line: {report.drop_caps_merged}")
    lines.append(f"  letter-spaced-caps normalized: {report.letter_spaced_normalized}")
    return "\n".join(lines)
