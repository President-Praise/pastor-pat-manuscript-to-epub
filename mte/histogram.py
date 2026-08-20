"""Font-size tier detection.

We build a histogram of font sizes weighted by character count, then discover
the tier structure. Body text is always the mode (most characters). Anything
noticeably larger than body is a heading tier candidate; we cluster those.

Output tiers, from smallest to largest:
    body
    h3            (subsection heading, small)
    h2            (section heading)
    chapter       (chapter title)
    part          (part title)

If the document uses fewer tiers, some slots will be empty. The classifier
only relies on comparisons against the body tier and the largest observed
tiers, so missing tiers are fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .pdf_extract import Line, Span


@dataclass
class TierMap:
    body: float
    tiers: list[float]  # sorted ascending, unique, includes body
    labels: dict[float, str]  # size -> tier label

    def label_for(self, size: float) -> str:
        # Snap to the nearest known tier size within 0.3 pt tolerance.
        best = None
        best_d = 1e9
        for t in self.tiers:
            d = abs(size - t)
            if d < best_d:
                best_d = d
                best = t
        if best is None or best_d > 0.3:
            # Not a known tier; label relative to body.
            if size <= self.body + 0.4:
                return "body"
            return "unknown-large"
        return self.labels[best]


def size_histogram(spans: Iterable[Span]) -> dict[float, int]:
    """Character-weighted histogram of span sizes."""
    hist: dict[float, int] = {}
    for s in spans:
        n = max(1, len(s.text.strip()))
        if not s.text.strip():
            continue
        hist[s.size] = hist.get(s.size, 0) + n
    return hist


def line_size_histogram(lines: Iterable[Line]) -> dict[float, int]:
    """Histogram over dominant-per-line sizes, weighted by line char count."""
    hist: dict[float, int] = {}
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        size = line.dominant_size
        hist[size] = hist.get(size, 0) + len(text)
    return hist


def detect_tiers(hist: dict[float, int]) -> TierMap:
    """Cluster the histogram into named tiers.

    Strategy:
      1. Body = size with the greatest character weight.
      2. Anything <= body + 0.4 pt is body (handles rounding noise).
      3. Above body: cluster sizes into groups where sizes within 0.6 pt of
         each other are the same tier; take the character-weighted centroid
         of each cluster as the tier size.
      4. Assign labels from the top down: largest cluster = "part",
         next = "chapter", next = "h2", rest = "h3".
    """
    if not hist:
        return TierMap(body=0.0, tiers=[], labels={})

    body = max(hist, key=lambda s: hist[s])

    # Group sizes above body into clusters.
    above = sorted(s for s in hist if s > body + 0.4)
    clusters: list[list[float]] = []
    for s in above:
        if clusters and s - clusters[-1][-1] <= 0.6:
            clusters[-1].append(s)
        else:
            clusters.append([s])

    # Reduce each cluster to a representative size (character-weighted mean).
    cluster_sizes: list[float] = []
    for cl in clusters:
        total_w = sum(hist[s] for s in cl)
        if total_w == 0:
            cluster_sizes.append(cl[0])
        else:
            centroid = sum(s * hist[s] for s in cl) / total_w
            cluster_sizes.append(round(centroid, 1))

    cluster_sizes.sort()

    # Assign labels top-down.
    labels_by_size: dict[float, str] = {body: "body"}
    tier_names_top_down = ["part", "chapter", "h2", "h3"]
    for i, size in enumerate(reversed(cluster_sizes)):
        if i < len(tier_names_top_down):
            labels_by_size[size] = tier_names_top_down[i]
        else:
            labels_by_size[size] = "h3"

    # Also snap all raw sizes in `cl` to their cluster centroid label for
    # ease of lookup by the classifier.
    for cl, centroid in zip(clusters, cluster_sizes):
        for raw in cl:
            if raw not in labels_by_size:
                labels_by_size[raw] = labels_by_size[centroid]

    tiers_sorted = sorted(set([body] + cluster_sizes))
    return TierMap(body=body, tiers=tiers_sorted, labels=labels_by_size)


def format_histogram_report(hist: dict[float, int], tiers: TierMap) -> str:
    """Human-readable histogram + tier assignment."""
    total = sum(hist.values()) or 1
    lines = ["Font-size histogram (character-weighted):"]
    lines.append(f"  {'size':>6}  {'chars':>10}  {'%':>6}  tier")
    for size in sorted(hist, reverse=True):
        chars = hist[size]
        pct = 100.0 * chars / total
        label = tiers.label_for(size)
        lines.append(f"  {size:>6.1f}  {chars:>10d}  {pct:>5.1f}%  {label}")
    lines.append("")
    lines.append("Detected tier sizes:")
    for t in sorted(tiers.tiers, reverse=True):
        lines.append(f"  {tiers.labels.get(t, '?'):>8}: {t:.1f} pt")
    return "\n".join(lines)
