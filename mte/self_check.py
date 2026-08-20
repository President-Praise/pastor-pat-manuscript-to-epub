"""Post-write assertions on a written EPUB.

Reads the EPUB back and verifies five properties:

    1. Every <li> has non-whitespace text content.
    2. <table> count in each chapter XHTML matches Table blocks in the IR.
    3. Every chapter file's first heading is <h1>; a preceding
       <p class="chapter-caption"> exists iff the section has caption.
    4. No heading matches the punchline pattern
       (bold-caps-centered-looking, ends with . or !, <90 chars).
    5. <ol start="N"> continuity — start attributes match IR item_starts().

Returns a report of successes and failures per section.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .ir import Block, Book, Callout, ListBlock, Section, Table
from .epub_writer import WrittenEpub, WrittenSection


@dataclass
class CheckResult:
    section_slug: str
    check: str
    passed: bool
    detail: str = ""


@dataclass
class SelfCheckReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    @property
    def passed(self) -> bool:
        return not self.failures

    def format(self) -> str:
        lines = []
        by_slug: dict[str, list[CheckResult]] = {}
        for r in self.results:
            by_slug.setdefault(r.section_slug, []).append(r)
        for slug in sorted(by_slug):
            for r in by_slug[slug]:
                mark = "PASS" if r.passed else "FAIL"
                detail = f"  {r.detail}" if r.detail else ""
                lines.append(f"  [{mark}] {slug} :: {r.check}{detail}")
        return "\n".join(lines)


_LI_TEXT_RE = re.compile(r"<li\b[^>]*>(.*?)</li>", re.DOTALL | re.IGNORECASE)
_TABLE_RE = re.compile(r"<table\b", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"<(h[1-6])\b[^>]*>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_CHAPTER_CAPTION_RE = re.compile(
    r'<p[^>]*class="[^"]*chapter-caption[^"]*"[^>]*>(.*?)</p>',
    re.DOTALL | re.IGNORECASE,
)
_OL_START_RE = re.compile(r'<ol\b([^>]*)>', re.IGNORECASE)
_START_ATTR_RE = re.compile(r'start="(\d+)"', re.IGNORECASE)


def _strip_tags(html: str) -> str:
    return _TAG_RE.sub("", html)


def _count_blocks_by_type(blocks: list[Block], target: type) -> int:
    n = 0
    for b in blocks:
        if isinstance(b, target):
            n += 1
        if isinstance(b, Callout):
            n += _count_blocks_by_type(b.content, target)
        if isinstance(b, ListBlock):
            for seg in b.segments:
                from .ir import InterruptionSegment
                if isinstance(seg, InterruptionSegment):
                    n += _count_blocks_by_type(seg.blocks, target)
    return n


def _collect_list_blocks(blocks: list[Block]) -> list[ListBlock]:
    out: list[ListBlock] = []
    for b in blocks:
        if isinstance(b, ListBlock):
            out.append(b)
        if isinstance(b, Callout):
            out.extend(_collect_list_blocks(b.content))
        if isinstance(b, ListBlock):
            from .ir import InterruptionSegment
            for seg in b.segments:
                if isinstance(seg, InterruptionSegment):
                    out.extend(_collect_list_blocks(seg.blocks))
    return out


def _check_list_items_nonempty(sec: Section, xhtml: str) -> CheckResult:
    for m in _LI_TEXT_RE.finditer(xhtml):
        inner = _strip_tags(m.group(1)).strip()
        if not inner:
            return CheckResult(
                sec.slug, "list-items-nonempty", False,
                f"empty <li> found: {m.group(0)[:80]}",
            )
    return CheckResult(sec.slug, "list-items-nonempty", True)


def _check_table_count(sec: Section, xhtml: str) -> CheckResult:
    xhtml_count = len(_TABLE_RE.findall(xhtml))
    ir_count = _count_blocks_by_type(sec.content, Table)
    ok = xhtml_count == ir_count
    return CheckResult(
        sec.slug, "table-count-matches-ir", ok,
        "" if ok else f"xhtml={xhtml_count} ir={ir_count}",
    )


def _check_chapter_title(sec: Section, xhtml: str) -> CheckResult:
    m = _HEADING_RE.search(xhtml)
    if not m:
        return CheckResult(sec.slug, "chapter-title-h1", False, "no headings in file")
    if m.group(1).lower() != "h1":
        return CheckResult(
            sec.slug, "chapter-title-h1", False,
            f"first heading is <{m.group(1)}>, expected <h1>",
        )

    cap_match = _CHAPTER_CAPTION_RE.search(xhtml[:m.start()])
    has_caption_html = cap_match is not None
    has_caption_ir = sec.caption is not None
    if has_caption_html != has_caption_ir:
        return CheckResult(
            sec.slug, "chapter-title-h1", False,
            f"caption mismatch: ir={has_caption_ir} html={has_caption_html}",
        )
    return CheckResult(sec.slug, "chapter-title-h1", True)


_TERMINAL_PUNCT = (".", "!")


def _check_no_punchline_headings(sec: Section, xhtml: str) -> CheckResult:
    for m in _HEADING_RE.finditer(xhtml):
        tag = m.group(1).lower()
        if tag not in ("h2", "h3"):
            continue
        text = _strip_tags(m.group(2)).strip()
        if not text or len(text) >= 90:
            continue
        if not text.endswith(_TERMINAL_PUNCT):
            continue
        letters = [c for c in text if c.isalpha()]
        if not letters:
            continue
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.85:
            return CheckResult(
                sec.slug, "no-punchline-headings", False,
                f"suspect <{tag}>: {text[:80]}",
            )
    return CheckResult(sec.slug, "no-punchline-headings", True)


def _check_ol_starts(sec: Section, xhtml: str) -> CheckResult:
    lb_starts_in_ir: list[list[int]] = []
    ir_lists = _collect_list_blocks(sec.content)
    for lb in ir_lists:
        if lb.style != "ordered":
            continue
        lb_starts_in_ir.append(lb.item_starts())

    html_ol_starts: list[int] = []
    for m in _OL_START_RE.finditer(xhtml):
        sm = _START_ATTR_RE.search(m.group(1))
        html_ol_starts.append(int(sm.group(1)) if sm else 1)

    ir_flat: list[int] = [s for run in lb_starts_in_ir for s in run]
    if len(ir_flat) != len(html_ol_starts):
        return CheckResult(
            sec.slug, "ol-continuity", False,
            f"<ol> count mismatch: html={len(html_ol_starts)} ir={len(ir_flat)}",
        )
    for i, (ir_s, html_s) in enumerate(zip(ir_flat, html_ol_starts)):
        if ir_s != html_s:
            return CheckResult(
                sec.slug, "ol-continuity", False,
                f"<ol>#{i}: ir start={ir_s} html start={html_s}",
            )
    return CheckResult(sec.slug, "ol-continuity", True)


_SECTION_CHECKS = (
    _check_list_items_nonempty,
    _check_table_count,
    _check_chapter_title,
    _check_no_punchline_headings,
    _check_ol_starts,
)


def run_self_checks(book: Book, epub: WrittenEpub) -> SelfCheckReport:
    """Validate every written section (including synthetic Part landings)."""
    report = SelfCheckReport()

    with zipfile.ZipFile(epub.output_path, "r") as zf:
        for w in epub.section_files:
            xhtml = zf.read(f"OEBPS/{w.href}").decode("utf-8")
            for check in _SECTION_CHECKS:
                report.results.append(check(w.section, xhtml))

    return report
