#!/usr/bin/env python3
"""manuscript_to_epub: convert PDF/DOCX manuscripts to Kindle-ready EPUB.

Subcommands:
  dry-run  Extract, classify, and print histogram + outline. No EPUB.
  build    Extract, classify, and write an EPUB. Runs self-checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mte.pdf_extract import extract_spans, group_into_lines
from mte.histogram import line_size_histogram, format_histogram_report, detect_tiers
from mte.preprocess import preprocess, format_strip_report
from mte.tiers import assign_tiers, format_tier_report
from mte.classify import classify, outline_summary, format_outline
from mte.build_ir import build_book
from mte.ir import BookMetadata
from mte.epub_writer import write_epub
from mte.self_check import run_self_checks


def _load_lines(input_path: Path):
    ext = input_path.suffix.lower()
    if ext == ".pdf":
        spans = extract_spans(str(input_path))
        lines = group_into_lines(spans)
        return spans, lines
    if ext == ".docx":
        raise NotImplementedError("DOCX ingester not yet implemented.")
    raise SystemExit(f"Unsupported input type: {ext}")


def cmd_dry_run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2

    print(f"Loading {input_path.name} ...", file=sys.stderr)
    spans, lines = _load_lines(input_path)
    print(f"  {len(spans)} spans, {len(lines)} visual lines (raw)", file=sys.stderr)

    kept, strip_report = preprocess(lines)
    print(f"  {len(kept)} lines after preprocess", file=sys.stderr)

    print()
    print(format_strip_report(strip_report))
    print()

    tiers = assign_tiers(kept)
    print(format_tier_report(tiers))
    print()

    items = classify(kept, tiers)
    print("Classified outline:")
    print(format_outline(items))
    print()
    counts = outline_summary(items)
    print("Label counts:")
    for k in sorted(counts):
        print(f"  {k:>18}: {counts[k]}")

    low_conf = [c for c in items if c.confidence < 1.0 and c.label != "body"]
    if low_conf:
        print()
        print(f"Low-confidence heading classifications ({len(low_conf)}):")
        for c in low_conf[:20]:
            print(f"  [p{c.line.page:>3} {c.line.dominant_size:>4.1f}pt {c.label:>15} conf={c.confidence:.1f}]  {c.text[:70]}")
            for note in c.notes:
                print(f"      note: {note}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2

    print(f"Loading {input_path.name} ...", file=sys.stderr)
    spans, lines = _load_lines(input_path)
    print(f"  {len(spans)} spans, {len(lines)} raw lines", file=sys.stderr)

    kept, _ = preprocess(lines)
    print(f"  {len(kept)} lines after preprocess", file=sys.stderr)

    tiers = assign_tiers(kept)
    items = classify(kept, tiers)
    print(f"  {len(items)} classified items", file=sys.stderr)

    metadata = BookMetadata(
        title=args.title, author=args.author, language=args.language,
    )
    book = build_book(items, metadata)
    n_sections = sum(1 for _ in book.walk_sections())
    print(f"  IR: {len(book.front_matter)} front, {len(book.parts)} parts, "
          f"{len(book.chapters)} loose chapters, {len(book.back_matter)} back "
          f"({n_sections} total sections)", file=sys.stderr)

    epub = write_epub(book, args.output)
    print(f"  wrote {epub.output_path} ({len(epub.section_files)} section files)", file=sys.stderr)

    report = run_self_checks(book, epub)
    print(file=sys.stderr)
    print("Self-check results:", file=sys.stderr)
    if report.passed:
        print(f"  All {len(report.results)} checks passed.", file=sys.stderr)
    else:
        print(f"  {len(report.failures)} of {len(report.results)} checks FAILED:", file=sys.stderr)
        for f in report.failures[:40]:
            print(f"    FAIL [{f.section_slug}] {f.check}: {f.detail}", file=sys.stderr)
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dry = sub.add_parser("dry-run", help="Show histogram + detected outline, no EPUB")
    p_dry.add_argument("input", help="Path to .pdf or .docx")
    p_dry.set_defaults(func=cmd_dry_run)

    p_build = sub.add_parser("build", help="Generate EPUB")
    p_build.add_argument("input", help="Path to .pdf or .docx")
    p_build.add_argument("--title", required=True)
    p_build.add_argument("--author", required=True)
    p_build.add_argument("--output", required=True, help="Output .epub path")
    p_build.add_argument("--language", default="en")
    p_build.set_defaults(func=cmd_build)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
