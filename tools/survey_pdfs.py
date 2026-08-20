#!/usr/bin/env python3
"""Survey each PDF: histogram + samples per candidate heading tier + alignment/bold flags.

Purpose: check whether the six sample manuscripts share a common design language
before locking in a classifier. We DO NOT want tier detection that only works
for one book.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mte.pdf_extract import extract_spans, group_into_lines
from mte.histogram import line_size_histogram


def is_letter_spaced_caps(text: str) -> bool:
    t = text.strip()
    if len(t) < 5:
        return False
    # Pattern: single caps letters (or single chars) separated by spaces.
    # Roughly: > 40% of chars are spaces AND alpha chars are mostly uppercase.
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    space_ratio = t.count(" ") / len(t)
    return upper_ratio > 0.8 and space_ratio > 0.35


def all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) > 0.9


def survey(pdf_path: Path, max_samples: int = 5) -> None:
    print("=" * 78)
    print(pdf_path.name)
    print("=" * 78)
    spans = extract_spans(str(pdf_path))
    lines = group_into_lines(spans)
    pages = max((s.page for s in spans), default=0)
    print(f"pages: {pages}   spans: {len(spans)}   lines: {len(lines)}")

    hist = line_size_histogram(lines)
    if not hist:
        print("(empty)")
        return
    body = max(hist, key=lambda s: hist[s])
    total = sum(hist.values())
    print(f"\nbody size (most chars): {body:.1f} pt  ({100*hist[body]/total:.1f}% of chars)")

    print("\nsize  chars    %    role-guess")
    for size in sorted(hist, reverse=True):
        n = hist[size]
        pct = 100 * n / total
        role = "body" if abs(size - body) < 0.4 else (
            "sub-body" if size < body else "candidate-heading"
        )
        print(f"{size:>5.1f}  {n:>6d}  {pct:>4.1f}%   {role}")

    # For each candidate heading tier, show up to N sample lines with flags.
    print("\ncandidate-heading tiers (samples):")
    heading_sizes = sorted((s for s in hist if s > body + 0.4), reverse=True)
    for size in heading_sizes:
        samples = []
        for ln in lines:
            if abs(ln.dominant_size - size) < 0.05:
                text = ln.text.strip()
                if not text:
                    continue
                flags = []
                if ln.is_centered: flags.append("CTR")
                if ln.bold_ratio > 0.5: flags.append("BOLD")
                if ln.italic_ratio > 0.5: flags.append("IT")
                if is_letter_spaced_caps(text): flags.append("LSPACE-CAPS")
                elif all_caps(text): flags.append("CAPS")
                display = text if len(text) <= 60 else text[:57] + "..."
                samples.append(f"    p{ln.page:>3}  [{','.join(flags):<20}]  {display}")
                if len(samples) >= max_samples:
                    break
        print(f"\n  size {size:.1f} pt  ({hist[size]} chars, ~{hist[size]/max(1,size):.0f} chars-per-pt-weight):")
        for s in samples:
            print(s)

    # Also — quick footer/header detection: find bare-numeric lines and near-page-edge lines that repeat.
    print("\nrepeated header/footer candidates (lines appearing on many pages, similar y):")
    from collections import Counter
    fingerprints: Counter = Counter()
    fp_examples: dict[tuple, str] = {}
    for ln in lines:
        text = ln.text.strip()
        if not text or len(text) > 80:
            continue
        # bin y0 to 5-pt buckets and top-or-bottom quintile
        y_frac = ln.y0 / max(1, ln.page_height)
        if 0.10 < y_frac < 0.90:
            continue  # only look at top/bottom margins
        # normalize: replace any digit run with #
        import re
        norm = re.sub(r"\d+", "#", text)
        key = (norm, "TOP" if y_frac < 0.5 else "BOT")
        fingerprints[key] += 1
        fp_examples.setdefault(key, text)
    for (norm, pos), count in fingerprints.most_common(8):
        if count < 3:
            continue
        example = fp_examples.get((norm, pos), "")
        print(f"  {pos}  x{count:>3}  norm='{norm[:40]}'   ex='{example[:40]}'")


def main() -> int:
    d = Path(__file__).resolve().parents[1] / "001"
    pdfs = sorted(d.glob("*.pdf"))
    if not pdfs:
        print("no PDFs found", file=sys.stderr)
        return 1
    for pdf in pdfs:
        try:
            survey(pdf)
        except Exception as e:
            print(f"error on {pdf.name}: {e}", file=sys.stderr)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
