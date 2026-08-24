"""Verify a generated EPUB against its source manuscript.

Two independent checks:

    Layer 1 (structural)
        Reduce both the in-memory IR and the on-disk EPUB to the same
        canonical [(section, kind, text)] block stream, then diff.
        Catches serialization bugs: dropped/duplicated/reordered blocks,
        glued or fractured list items, wrong CSS routing.

    Layer 3 (raw text, classifier-independent)
        Flat text extraction of the PDF via PyMuPDF's get_text("text")
        (not the font-metric rawdict path used by classify()), plus
        header/footer stripping and letter-spacing collapse. Diff against
        the EPUB's flattened text. Catches classification bugs that
        Layer 1 is structurally blind to. Expected to be noisier — use as
        an eyeball spot-check, not a gate.

Both checks emit a terminal summary and an HTML report under out/verify/.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF — already a dependency of pdf_extract
from lxml import html as lxml_html

from .ir import (
    Block, Book, Callout, Heading, InterruptionSegment, ItemSegment,
    ListBlock, Paragraph, PullQuote, Section, SceneBreak, Table, TextRun,
)
from .epub_writer import WrittenEpub, _collect_sections


# --------------------------------------------------------------------------- #
# Canonical block stream
# --------------------------------------------------------------------------- #

# Kinds mirror the IR block types collapsed to what an EPUB reader can
# meaningfully distinguish. Deliberately coarse: e.g. all callout content
# rolls up as "callout_body" regardless of variant, because a wrong
# variant is a styling bug, not a content-drop bug (which is what Layer 1
# targets).
BLOCK_KINDS = frozenset({
    "chapter_caption",
    "chapter_title",
    "h2",
    "h3",
    "body",
    "pull_quote",
    "scene_break",
    "list_item_ordered",
    "list_item_unordered",
    "callout_body",
    "table_row",
})


@dataclass(frozen=True)
class StreamBlock:
    section_slug: str
    kind: str
    text: str  # normalized

    def key(self) -> tuple[str, str]:
        """Diff key: (kind, text[:120]) — trimmed so long paragraphs still align."""
        return (self.kind, self.text[:120])


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #

_SOFT_HYPHEN = "­"
_SMART_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
}
_DASHES = {
    "–": "-", "—": "-", "−": "-",
}
_WS_RE = re.compile(r"\s+")
# 3+ single-letter tokens separated by single spaces → glue them:
# "A N A G K A Z Ō" -> "ANAGKAZŌ". Only fires when every token is 1 letter
# so we don't clobber "I am a" prose.
_LETTER_SPACED_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))((?:[^\W\d_]\s){2,}[^\W\d_])(?=\s|$)",
    re.UNICODE,
)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_SOFT_HYPHEN, "")
    for src, dst in _SMART_QUOTES.items():
        text = text.replace(src, dst)
    for src, dst in _DASHES.items():
        text = text.replace(src, dst)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _collapse_letter_spacing(text: str) -> str:
    """Fold 'A N A G K A Z Ō' → 'ANAGKAZŌ'. Layer-3 only — Layer 1 skips this
    because classify() has already handled decorative titles consistently."""
    def _glue(m: re.Match) -> str:
        return m.group(1).replace(" ", "")
    return _LETTER_SPACED_RE.sub(_glue, text)


# --------------------------------------------------------------------------- #
# IR → stream
# --------------------------------------------------------------------------- #

def _runs_text(runs: list[TextRun]) -> str:
    return normalize_text("".join(r.text for r in runs))


def _emit_block(sec_slug: str, block: Block, out: list[StreamBlock]) -> None:
    if isinstance(block, Heading):
        kind = "h2" if block.level == 2 else "h3"
        out.append(StreamBlock(sec_slug, kind, _runs_text(block.runs)))
        return
    if isinstance(block, Paragraph):
        out.append(StreamBlock(sec_slug, "body", _runs_text(block.runs)))
        return
    if isinstance(block, PullQuote):
        out.append(StreamBlock(sec_slug, "pull_quote", _runs_text(block.runs)))
        if block.attribution:
            out.append(StreamBlock(
                sec_slug, "pull_quote",
                normalize_text(f"- {block.attribution}"),
            ))
        return
    if isinstance(block, SceneBreak):
        out.append(StreamBlock(sec_slug, "scene_break", ""))
        return
    if isinstance(block, ListBlock):
        kind = "list_item_ordered" if block.style == "ordered" else "list_item_unordered"
        for seg in block.segments:
            if isinstance(seg, ItemSegment):
                for item in seg.items:
                    out.append(StreamBlock(sec_slug, kind, _runs_text(item.runs)))
                    for child in item.children:
                        _emit_block(sec_slug, child, out)
            elif isinstance(seg, InterruptionSegment):
                for b in seg.blocks:
                    _emit_block(sec_slug, b, out)
        return
    if isinstance(block, Table):
        for row in block.header_rows + block.body_rows:
            cell_texts: list[str] = []
            for cell in row.cells:
                sub: list[StreamBlock] = []
                for b in cell.blocks:
                    _emit_block(sec_slug, b, sub)
                cell_texts.append(" | ".join(s.text for s in sub if s.text))
            out.append(StreamBlock(
                sec_slug, "table_row", normalize_text(" || ".join(cell_texts)),
            ))
        return
    if isinstance(block, Callout):
        for b in block.content:
            sub: list[StreamBlock] = []
            _emit_block(sec_slug, b, sub)
            for s in sub:
                out.append(StreamBlock(s.section_slug, "callout_body", s.text))
        return
    raise TypeError(f"unknown block type: {type(block).__name__}")


def stream_from_ir(book: Book) -> list[StreamBlock]:
    """Reproduce the exact section list the writer serializes (including
    synthetic Part landing sections) and emit a canonical block stream.

    The visible TOC section is skipped: the writer regenerates its body from
    the Book outline (so entries are clickable), so diffing extracted-from-PDF
    against regenerated-from-IR content is not meaningful."""
    from .epub_writer import is_toc_section
    out: list[StreamBlock] = []
    for sec in _collect_sections(book):
        if is_toc_section(sec):
            continue
        if sec.caption:
            out.append(StreamBlock(
                sec.slug, "chapter_caption", normalize_text(sec.caption),
            ))
        out.append(StreamBlock(
            sec.slug, "chapter_title", normalize_text(sec.title),
        ))
        for b in sec.content:
            _emit_block(sec.slug, b, out)
    return out


# --------------------------------------------------------------------------- #
# EPUB → stream
# --------------------------------------------------------------------------- #

_OPF_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "cnt": "urn:oasis:names:tc:opendocument:xmlns:container",
}


def _read_epub_spine(epub_path: Path) -> list[tuple[str, str]]:
    """Return [(slug, xhtml_content)] in spine order, skipping cover/nav.

    slug is derived from the href basename (strip 'text/' and '.xhtml').
    """
    from lxml import etree
    with zipfile.ZipFile(epub_path, "r") as zf:
        container = etree.fromstring(zf.read("META-INF/container.xml"))
        opf_path = container.find(".//cnt:rootfile", _OPF_NS).get("full-path")
        opf_dir = Path(opf_path).parent
        opf = etree.fromstring(zf.read(opf_path))
        manifest = {
            item.get("id"): item.get("href")
            for item in opf.findall(".//opf:manifest/opf:item", _OPF_NS)
        }
        spine_ids = [
            ref.get("idref")
            for ref in opf.findall(".//opf:spine/opf:itemref", _OPF_NS)
        ]
        out: list[tuple[str, str]] = []
        for sid in spine_ids:
            href = manifest.get(sid)
            if not href:
                continue
            if sid in ("cover-page", "nav"):
                continue
            slug = Path(href).stem
            full = str(opf_dir / href) if str(opf_dir) != "." else href
            xhtml = zf.read(full).decode("utf-8")
            out.append((slug, xhtml))
    return out


def _text_of(el) -> str:
    return normalize_text(el.text_content() if hasattr(el, "text_content") else "")


def _emit_from_xhtml_node(slug: str, node, out: list[StreamBlock], *,
                          inside_callout: bool = False) -> None:
    tag = node.tag.lower() if isinstance(node.tag, str) else None
    if tag is None:
        return
    classes = (node.get("class") or "").split()

    if tag == "h1":
        # h1 is handled at chapter level by the caller
        return
    if tag == "h2":
        out.append(StreamBlock(slug, "h2", _text_of(node)))
        return
    if tag == "h3":
        out.append(StreamBlock(slug, "h3", _text_of(node)))
        return
    if tag == "p":
        if "chapter-caption" in classes:
            # Handled at chapter level
            return
        if "subtitle-tagline" in classes or "italic-quote" in classes:
            out.append(StreamBlock(slug, "pull_quote", _text_of(node)))
            return
        if "attribution" in classes:
            out.append(StreamBlock(
                slug, "pull_quote", normalize_text("- " + (node.text_content() or "").lstrip("-— ").strip()),
            ))
            return
        kind = "callout_body" if inside_callout else "body"
        out.append(StreamBlock(slug, kind, _text_of(node)))
        return
    if tag == "hr":
        if "scene-break" in classes:
            out.append(StreamBlock(slug, "scene_break", ""))
        return
    if tag in ("ol", "ul"):
        kind = "list_item_ordered" if tag == "ol" else "list_item_unordered"
        for child in node:
            if isinstance(child.tag, str) and child.tag.lower() == "li":
                # Item's own runs = direct text + inline children, excluding nested block children
                own = _li_own_text(child)
                out.append(StreamBlock(slug, kind, normalize_text(own)))
                for sub in child:
                    if isinstance(sub.tag, str) and sub.tag.lower() in (
                        "ol", "ul", "p", "table", "aside", "blockquote", "hr", "h2", "h3"
                    ):
                        _emit_from_xhtml_node(slug, sub, out, inside_callout=inside_callout)
        return
    if tag == "table":
        for row in node.iter("tr"):
            cell_texts: list[str] = []
            for cell in row:
                if isinstance(cell.tag, str) and cell.tag.lower() in ("td", "th"):
                    cell_texts.append(_text_of(cell))
            out.append(StreamBlock(
                slug, "table_row", normalize_text(" || ".join(cell_texts)),
            ))
        return
    if tag == "blockquote":
        # scripture / epigraph blockquotes are pull-quote holders in the IR.
        # Their <p> children (text + attribution) map to pull_quote entries.
        for child in node:
            ctag = child.tag.lower() if isinstance(child.tag, str) else ""
            cclasses = (child.get("class") or "").split()
            if ctag != "p":
                _emit_from_xhtml_node(slug, child, out)
                continue
            if "attribution" in cclasses:
                text = (child.text_content() or "").lstrip("-— ").strip()
                out.append(StreamBlock(slug, "pull_quote", normalize_text("- " + text)))
            else:
                out.append(StreamBlock(slug, "pull_quote", _text_of(child)))
        return
    if tag == "aside":
        for child in node:
            _emit_from_xhtml_node(slug, child, out, inside_callout=True)
        return
    # Anything else: descend
    for child in node:
        _emit_from_xhtml_node(slug, child, out, inside_callout=inside_callout)


_INLINE_TAGS = {"strong", "em", "b", "i", "span", "a", "sup", "sub", "small"}


def _li_own_text(li_el) -> str:
    """Text of an <li> excluding nested block children (nested lists, tables, etc.)."""
    parts: list[str] = []
    if li_el.text:
        parts.append(li_el.text)
    for child in li_el:
        tag = child.tag.lower() if isinstance(child.tag, str) else ""
        if tag in _INLINE_TAGS:
            parts.append(child.text_content() or "")
        # nested block children contribute nothing to the item's own text
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def stream_from_epub(epub_path: Path) -> list[StreamBlock]:
    from .epub_writer import _TOC_TITLE_PATTERNS
    out: list[StreamBlock] = []
    for slug, xhtml in _read_epub_spine(epub_path):
        # lxml.html is namespace-tolerant and strips xhtml xmlns cleanly.
        # Feed bytes so the XML encoding declaration doesn't upset lxml.
        root = lxml_html.fromstring(xhtml.encode("utf-8"))
        body = root.find(".//body")
        if body is None:
            continue

        # Chapter caption + title come first in the section
        caption_el = body.find(".//p[@class='chapter-caption']")
        h1 = body.find(".//h1")
        # Skip the visible TOC page — its body is regenerated from the outline
        # and won't match the IR's extracted-from-PDF content.
        if h1 is not None and _text_of(h1).strip().lower() in _TOC_TITLE_PATTERNS:
            continue
        if caption_el is not None:
            out.append(StreamBlock(
                slug, "chapter_caption", _text_of(caption_el),
            ))
        if h1 is not None:
            out.append(StreamBlock(slug, "chapter_title", _text_of(h1)))

        # Walk body in document order, skipping the caption + h1 we already emitted
        skip = set()
        if caption_el is not None:
            skip.add(id(caption_el))
        if h1 is not None:
            skip.add(id(h1))
        for child in body:
            if id(child) in skip:
                continue
            _emit_from_xhtml_node(slug, child, out)
    return out


# --------------------------------------------------------------------------- #
# Stray-capital corruption detector
# --------------------------------------------------------------------------- #
#
# Regression guard for the drop-cap merge false-positive class of bug:
# `_merge_split_initial_caps` (and, in principle, `_merge_drop_caps`) can
# glue drop-cap letters from a decorative title onto the words of the
# following subtitle, producing corrupted words like "Kand", "IPriestly",
# "NIdentity", "Sof", "Athe", "WSaints", "AMust". These strings survive
# Layer 1's structural diff (IR and EPUB agree — both are equally wrong)
# so we detect them by pattern instead.

# Two-capitals-then-lowercase: "NIdentity", "AMust", "WSaints", "IPriestly".
# Legitimate matches in prose are essentially zero — proper brands like
# "MicroUSB" or "iPhone" don't fit this exact anchor + shape.
_STRAY_CAP_DOUBLE_RE = re.compile(r"\b[A-Z][A-Z][a-z]+\b")

# Function words whose "[single capital] + word" concatenation does NOT
# collide with any common English word — so flagging them is safe.
# Excludes: and (→ Band/Hand/Land/Sand), or (→ For/Cor/Nor), in (→ Bin/
# Fin/Pin/Sin/Tin/Win), is (→ His/Wis), on (→ Con/Don/Ion/Son/Ton), an
# (→ Ban/Can/Dan/Fan/Man/Pan/Tan/Van), at (→ Bat/Cat/Mat/Rat/Sat/Pat),
# am (→ Ham/Jam/Ram/Sam), as (→ Has/Gas/Was), be (→ Abe, rare). All of
# these 2-letter suffixes collide with legitimate 3-letter English words
# when prefixed with any single capital, so pattern-matching them would
# fire on every "His"/"Son"/"For"/"Man" in prose.
_STRAY_TAIL_WORDS = ("the", "of", "to", "if", "by", "yet", "but", "into", "onto", "from")
# "Kand"-style: a single capital fused to a lowercase function word that
# doesn't produce a real English word ("Athe", "Sof", "Xto", "Xif").
_STRAY_CAP_FUNCTION_RE = re.compile(
    r"\b[A-Z](" + "|".join(sorted(_STRAY_TAIL_WORDS, key=len, reverse=True)) + r")\b"
)


@dataclass
class StrayCapFinding:
    section_slug: str
    kind: str          # block kind where it was found
    word: str          # the offending token as it appears
    context: str       # up to ~80 chars around the token for eyeballing


def find_stray_capitals(stream: list["StreamBlock"]) -> list[StrayCapFinding]:
    """Scan a block stream for stray-capital corruption tokens.

    Two patterns:
      1. Two consecutive capitals starting a Title-case word ("NIdentity").
      2. A single capital fused to a common lowercase function word ("Kand",
         "Sof", "Athe").
    Both are near-zero false-positive in normal English prose in this corpus
    (a hymn writer's manual would break rule 1; none of these books have
    that). Rule 2 explicitly excludes single-capital sentence-openers like
    "The"/"And" by requiring no space between the stray cap and the word.
    """
    findings: list[StrayCapFinding] = []
    for b in stream:
        text = b.text
        if not text:
            continue
        for m in _STRAY_CAP_DOUBLE_RE.finditer(text):
            findings.append(StrayCapFinding(
                section_slug=b.section_slug, kind=b.kind, word=m.group(0),
                context=_context_snippet(text, m.start(), m.end()),
            ))
        for m in _STRAY_CAP_FUNCTION_RE.finditer(text):
            findings.append(StrayCapFinding(
                section_slug=b.section_slug, kind=b.kind, word=m.group(0),
                context=_context_snippet(text, m.start(), m.end()),
            ))
    return findings


def _context_snippet(text: str, start: int, end: int, radius: int = 30) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(text) else ""
    return prefix + text[lo:hi] + suffix


# --------------------------------------------------------------------------- #
# Diff engine + report
# --------------------------------------------------------------------------- #

@dataclass
class SectionDiff:
    slug: str
    ir_count: int
    epub_count: int
    matched: int
    missing: list[StreamBlock] = field(default_factory=list)     # in IR, not in EPUB
    extra: list[StreamBlock] = field(default_factory=list)       # in EPUB, not in IR
    replaced: list[tuple[list[StreamBlock], list[StreamBlock]]] = field(default_factory=list)
    ir_body_sha: str = ""
    epub_body_sha: str = ""

    @property
    def ok(self) -> bool:
        return (
            not self.missing
            and not self.extra
            and not self.replaced
            and self.ir_body_sha == self.epub_body_sha
        )


@dataclass
class VerifyReport:
    section_diffs: list[SectionDiff] = field(default_factory=list)
    ir_total: int = 0
    epub_total: int = 0
    ir_wordcount: int = 0
    epub_wordcount: int = 0
    kind_counts_ir: dict[str, int] = field(default_factory=dict)
    kind_counts_epub: dict[str, int] = field(default_factory=dict)
    stray_caps: list[StrayCapFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            all(sd.ok for sd in self.section_diffs)
            and self._wordcount_ok()
            and not self.stray_caps
        )

    def _wordcount_ok(self) -> bool:
        if self.ir_wordcount == 0:
            return self.epub_wordcount == 0
        drift = abs(self.epub_wordcount - self.ir_wordcount) / self.ir_wordcount
        return drift <= 0.01

    def format(self) -> str:
        lines: list[str] = []
        lines.append(f"IR blocks: {self.ir_total}   EPUB blocks: {self.epub_total}")
        drift = 0.0
        if self.ir_wordcount:
            drift = 100 * abs(self.epub_wordcount - self.ir_wordcount) / self.ir_wordcount
        lines.append(
            f"Word count: IR={self.ir_wordcount:,}  EPUB={self.epub_wordcount:,}  "
            f"drift={drift:.2f}%"
        )
        all_kinds = sorted(set(self.kind_counts_ir) | set(self.kind_counts_epub))
        lines.append("Kind counts (IR / EPUB):")
        for k in all_kinds:
            i = self.kind_counts_ir.get(k, 0)
            e = self.kind_counts_epub.get(k, 0)
            mark = " " if i == e else "*"
            lines.append(f"  {mark} {k:>22}: {i:>5} / {e:<5}")
        lines.append("")
        for sd in self.section_diffs:
            status = "OK" if sd.ok else "DIFF"
            lines.append(
                f"[{status}] {sd.slug:<40} "
                f"ir={sd.ir_count:>4} epub={sd.epub_count:>4} "
                f"matched={sd.matched:>4} "
                f"missing={len(sd.missing):>3} extra={len(sd.extra):>3} "
                f"replaced={len(sd.replaced):>3}"
            )
            if not sd.ok:
                for m in sd.missing[:3]:
                    lines.append(f"    - MISSING  [{m.kind}]  {m.text[:90]}")
                for e in sd.extra[:3]:
                    lines.append(f"    + EXTRA    [{e.kind}]  {e.text[:90]}")
                for ir_run, ep_run in sd.replaced[:2]:
                    for b in ir_run[:2]:
                        lines.append(f"    ~ IR      [{b.kind}]  {b.text[:90]}")
                    for b in ep_run[:2]:
                        lines.append(f"    ~ EPUB    [{b.kind}]  {b.text[:90]}")
        if self.stray_caps:
            lines.append("")
            lines.append(f"STRAY CAPITALS ({len(self.stray_caps)} — likely drop-cap merge corruption):")
            for f in self.stray_caps[:20]:
                lines.append(
                    f"  ! [{f.section_slug}] [{f.kind}] {f.word!r}  in: {f.context}"
                )
            if len(self.stray_caps) > 20:
                lines.append(f"  … and {len(self.stray_caps) - 20} more")
        return "\n".join(lines)


def _sha1(items: Iterable[StreamBlock]) -> str:
    h = hashlib.sha1()
    for b in items:
        h.update(b.kind.encode("utf-8"))
        h.update(b"\x00")
        h.update(b.text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:12]


def _wordcount(items: Iterable[StreamBlock]) -> int:
    n = 0
    for b in items:
        n += len(b.text.split())
    return n


def _kind_counts(items: Iterable[StreamBlock]) -> dict[str, int]:
    out: dict[str, int] = {}
    for b in items:
        out[b.kind] = out.get(b.kind, 0) + 1
    return out


def _by_slug(items: list[StreamBlock]) -> dict[str, list[StreamBlock]]:
    out: dict[str, list[StreamBlock]] = {}
    for b in items:
        out.setdefault(b.section_slug, []).append(b)
    return out


def diff_streams(ir_stream: list[StreamBlock],
                 epub_stream: list[StreamBlock]) -> VerifyReport:
    report = VerifyReport(
        ir_total=len(ir_stream),
        epub_total=len(epub_stream),
        ir_wordcount=_wordcount(ir_stream),
        epub_wordcount=_wordcount(epub_stream),
        kind_counts_ir=_kind_counts(ir_stream),
        kind_counts_epub=_kind_counts(epub_stream),
    )
    ir_by = _by_slug(ir_stream)
    ep_by = _by_slug(epub_stream)
    all_slugs: list[str] = []
    seen: set[str] = set()
    for b in ir_stream + epub_stream:
        if b.section_slug not in seen:
            seen.add(b.section_slug)
            all_slugs.append(b.section_slug)

    for slug in all_slugs:
        ir_items = ir_by.get(slug, [])
        ep_items = ep_by.get(slug, [])
        sd = SectionDiff(
            slug=slug,
            ir_count=len(ir_items),
            epub_count=len(ep_items),
            matched=0,
            ir_body_sha=_sha1(ir_items),
            epub_body_sha=_sha1(ep_items),
        )
        sm = difflib.SequenceMatcher(
            a=[b.key() for b in ir_items],
            b=[b.key() for b in ep_items],
            autojunk=False,
        )
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                sd.matched += i2 - i1
            elif tag == "delete":
                sd.missing.extend(ir_items[i1:i2])
            elif tag == "insert":
                sd.extra.extend(ep_items[j1:j2])
            elif tag == "replace":
                sd.replaced.append((ir_items[i1:i2], ep_items[j1:j2]))
        report.section_diffs.append(sd)
    return report


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def _stray_caps_html(report: "VerifyReport") -> str:
    if not report.stray_caps:
        return ""
    rows = []
    for f in report.stray_caps:
        rows.append(
            f'<tr><td>{_html_escape(f.section_slug)}</td>'
            f'<td>{_html_escape(f.kind)}</td>'
            f'<td><code>{_html_escape(f.word)}</code></td>'
            f'<td>{_html_escape(f.context)}</td></tr>'
        )
    return (
        f'<h2>Stray capitals ({len(report.stray_caps)})</h2>'
        f'<p>Likely drop-cap merge corruption — a decorative title\'s letters '
        f'being spliced into the following subtitle\'s words.</p>'
        f'<table><tr><th>section</th><th>kind</th><th>token</th><th>context</th></tr>'
        f'{"".join(rows)}</table>'
    )


def write_html_report(report: VerifyReport, out_path: Path,
                      pdf_name: str, epub_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for sd in report.section_diffs:
        status = "ok" if sd.ok else "diff"
        rows.append(
            f'<tr class="{status}"><td>{_html_escape(sd.slug)}</td>'
            f'<td>{sd.ir_count}</td><td>{sd.epub_count}</td>'
            f'<td>{sd.matched}</td><td>{len(sd.missing)}</td>'
            f'<td>{len(sd.extra)}</td><td>{len(sd.replaced)}</td></tr>'
        )
        if not sd.ok:
            details: list[str] = []
            for m in sd.missing:
                details.append(
                    f'<div class="row missing">- <span class="k">[{m.kind}]</span> {_html_escape(m.text)}</div>'
                )
            for e in sd.extra:
                details.append(
                    f'<div class="row extra">+ <span class="k">[{e.kind}]</span> {_html_escape(e.text)}</div>'
                )
            for ir_run, ep_run in sd.replaced:
                details.append('<div class="row replaced">--- replaced ---</div>')
                for b in ir_run:
                    details.append(
                        f'<div class="row missing">~IR   <span class="k">[{b.kind}]</span> {_html_escape(b.text)}</div>'
                    )
                for b in ep_run:
                    details.append(
                        f'<div class="row extra">~EPUB <span class="k">[{b.kind}]</span> {_html_escape(b.text)}</div>'
                    )
            rows.append(
                f'<tr class="detail"><td colspan="7"><details><summary>diffs</summary>{"".join(details)}</details></td></tr>'
            )

    kind_rows: list[str] = []
    for k in sorted(set(report.kind_counts_ir) | set(report.kind_counts_epub)):
        i = report.kind_counts_ir.get(k, 0)
        e = report.kind_counts_epub.get(k, 0)
        cls = "" if i == e else "diff"
        kind_rows.append(f'<tr class="{cls}"><td>{k}</td><td>{i}</td><td>{e}</td></tr>')

    drift = 0.0
    if report.ir_wordcount:
        drift = 100 * abs(report.epub_wordcount - report.ir_wordcount) / report.ir_wordcount

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Verify: {_html_escape(epub_name)}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2em; }}
h1 {{ margin-bottom: 0.2em; }}
.meta {{ color: #666; margin-bottom: 1.5em; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
th {{ background: #f0f0f0; }}
tr.ok td {{ background: #f0fff0; }}
tr.diff td {{ background: #fff0f0; }}
tr.detail td {{ background: #fafafa; padding: 8px 12px; }}
.row {{ font-family: SF Mono, Menlo, monospace; font-size: 12px; margin: 2px 0; }}
.row.missing {{ color: #a00; }}
.row.extra {{ color: #060; }}
.row.replaced {{ color: #888; margin-top: 8px; }}
.k {{ color: #666; font-weight: 600; }}
summary {{ cursor: pointer; color: #06c; }}
</style></head><body>
<h1>Verify: {_html_escape(epub_name)}</h1>
<div class="meta">Source: {_html_escape(pdf_name)}<br>
IR blocks: {report.ir_total} &nbsp; EPUB blocks: {report.epub_total}<br>
Words: IR={report.ir_wordcount:,} &nbsp; EPUB={report.epub_wordcount:,} &nbsp; drift={drift:.2f}%<br>
<b>Overall: {"OK" if report.ok else "DIFFERENCES FOUND"}</b>
</div>
<h2>Kind counts</h2>
<table><tr><th>kind</th><th>IR</th><th>EPUB</th></tr>
{"".join(kind_rows)}
</table>
<h2>Per-section</h2>
<table><tr><th>section</th><th>IR</th><th>EPUB</th><th>matched</th><th>missing</th><th>extra</th><th>replaced</th></tr>
{"".join(rows)}
</table>
{_stray_caps_html(report)}
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Public entry: Layer 1
# --------------------------------------------------------------------------- #

def verify_structural(book: Book, epub_path: Path) -> VerifyReport:
    ir_stream = stream_from_ir(book)
    ep_stream = stream_from_epub(epub_path)
    report = diff_streams(ir_stream, ep_stream)
    # Stray-caps corruption is detected on the EPUB stream — that is what
    # the reader actually sees, and it's what we'd catch a regression in.
    report.stray_caps = find_stray_capitals(ep_stream)
    return report


# --------------------------------------------------------------------------- #
# Layer 3: raw pdf text vs flattened epub text
# --------------------------------------------------------------------------- #

def _raw_pdf_pages(pdf_path: Path) -> list[str]:
    """Plain (non-font-metric) text per page. Deliberately does NOT use the
    rawdict path that classify() ultimately consumes."""
    pages: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pages.append(page.get_text("text") or "")
    return pages


def _strip_repeating_headers_footers(pages: list[str],
                                     min_frequency: float = 0.3) -> list[str]:
    """Remove lines that appear at the top or bottom of >= min_frequency of pages."""
    if not pages:
        return pages
    top_counts: dict[str, int] = {}
    bot_counts: dict[str, int] = {}
    top_bot: list[tuple[list[str], list[str]]] = []
    for text in pages:
        raw_lines = [ln.strip() for ln in text.splitlines()]
        non_empty = [ln for ln in raw_lines if ln]
        top = non_empty[:2]
        bot = non_empty[-2:]
        top_bot.append((top, bot))
        for ln in top:
            top_counts[ln] = top_counts.get(ln, 0) + 1
        for ln in bot:
            bot_counts[ln] = bot_counts.get(ln, 0) + 1
    threshold = max(2, int(len(pages) * min_frequency))
    top_repeats = {ln for ln, n in top_counts.items() if n >= threshold}
    bot_repeats = {ln for ln, n in bot_counts.items() if n >= threshold}

    out: list[str] = []
    for text in pages:
        kept: list[str] = []
        raw_lines = text.splitlines()
        for i, ln in enumerate(raw_lines):
            stripped = ln.strip()
            if not stripped:
                kept.append(ln)
                continue
            # Only strip if this line appears near the start or end of the page
            if stripped in top_repeats and i < 3:
                continue
            if stripped in bot_repeats and i >= len(raw_lines) - 3:
                continue
            # Also filter bare page numbers on their own line
            if stripped.isdigit() and (i < 3 or i >= len(raw_lines) - 3):
                continue
            kept.append(ln)
        out.append("\n".join(kept))
    return out


def raw_pdf_text(pdf_path: Path) -> str:
    pages = _raw_pdf_pages(pdf_path)
    pages = _strip_repeating_headers_footers(pages)
    joined = "\n".join(pages)
    joined = _collapse_letter_spacing(joined)
    return normalize_text(joined)


def flatten_epub_text(epub_path: Path) -> str:
    parts: list[str] = []
    for _slug, xhtml in _read_epub_spine(epub_path):
        root = lxml_html.fromstring(xhtml.encode("utf-8"))
        body = root.find(".//body")
        if body is None:
            continue
        parts.append(body.text_content() or "")
    joined = "\n".join(parts)
    return normalize_text(joined)


@dataclass
class RawDiffReport:
    pdf_words: int
    epub_words: int
    matched_words: int
    missing_runs: list[str] = field(default_factory=list)   # in PDF, not EPUB
    extra_runs: list[str] = field(default_factory=list)     # in EPUB, not PDF

    @property
    def similarity(self) -> float:
        total = max(self.pdf_words, self.epub_words, 1)
        return self.matched_words / total

    def format(self, max_snippets: int = 20) -> str:
        lines: list[str] = []
        lines.append(
            f"Raw text: PDF={self.pdf_words:,} words  EPUB={self.epub_words:,} words  "
            f"similarity={self.similarity * 100:.1f}%"
        )
        lines.append(f"Missing runs (in PDF, not EPUB): {len(self.missing_runs)}")
        for r in self.missing_runs[:max_snippets]:
            lines.append(f"  - {r[:110]}")
        lines.append(f"Extra runs (in EPUB, not PDF): {len(self.extra_runs)}")
        for r in self.extra_runs[:max_snippets]:
            lines.append(f"  + {r[:110]}")
        lines.append(
            "\n(Layer 3 is expected to be noisy — TOC / front-matter layout "
            "produces false positives. Eyeball the diffs, don't gate on them.)"
        )
        return "\n".join(lines)


def raw_text_diff(pdf_path: Path, epub_path: Path,
                  min_run_words: int = 4) -> RawDiffReport:
    pdf_words = raw_pdf_text(pdf_path).split()
    epub_words = flatten_epub_text(epub_path).split()

    # Case-fold + strip punctuation for matching only; keep original words for display
    def _key(w: str) -> str:
        return re.sub(r"[^\w]", "", w).lower()

    pdf_keys = [_key(w) for w in pdf_words]
    epub_keys = [_key(w) for w in epub_words]

    sm = difflib.SequenceMatcher(a=pdf_keys, b=epub_keys, autojunk=False)
    matched = 0
    missing: list[str] = []
    extra: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
        elif tag == "delete":
            if (i2 - i1) >= min_run_words:
                missing.append(" ".join(pdf_words[i1:i2]))
        elif tag == "insert":
            if (j2 - j1) >= min_run_words:
                extra.append(" ".join(epub_words[j1:j2]))
        elif tag == "replace":
            if (i2 - i1) >= min_run_words:
                missing.append(" ".join(pdf_words[i1:i2]))
            if (j2 - j1) >= min_run_words:
                extra.append(" ".join(epub_words[j1:j2]))

    return RawDiffReport(
        pdf_words=len(pdf_words),
        epub_words=len(epub_words),
        matched_words=matched,
        missing_runs=missing,
        extra_runs=extra,
    )
