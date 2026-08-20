"""Serialize a Book IR into an EPUB3 archive.

Layout:
    mimetype                (STORED, first entry)
    META-INF/container.xml
    OEBPS/content.opf
    OEBPS/nav.xhtml         (EPUB3 nav doc)
    OEBPS/toc.ncx           (legacy NCX for older readers)
    OEBPS/styles.css
    OEBPS/text/<slug>.xhtml (one per Section)

Block dispatch is intentionally boring: each IR node maps 1:1 to a small
XHTML snippet. All escaping goes through xml.sax.saxutils.escape.
"""

from __future__ import annotations

import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape, quoteattr

from .ir import (
    Block, Book, Callout, Heading, InterruptionSegment, ItemSegment,
    ListBlock, Paragraph, PullQuote, Section, SceneBreak, Table, TextRun,
)


_CSS = """\
@charset "utf-8";

body { font-family: Georgia, "Times New Roman", serif; line-height: 1.5; margin: 0.6em 1em; }
h1, h2, h3 { font-family: Helvetica, Arial, sans-serif; text-align: center; }
h1 { font-size: 1.7em; margin: 1.6em 0 0.6em; }
h2 { font-size: 1.25em; margin: 1.4em 0 0.4em; }
h3 { font-size: 1.1em; margin: 1.1em 0 0.3em; }
p { margin: 0 0 0.6em; text-indent: 1.2em; }
p.no-indent, p.center, p.chapter-caption, h1 + p, h2 + p, h3 + p { text-indent: 0; }
p.center { text-align: center; }
p.chapter-caption { text-align: center; text-transform: uppercase; letter-spacing: 0.15em;
    font-size: 0.85em; margin-top: 2em; margin-bottom: 0.2em; }
hr.scene-break { border: none; text-align: center; margin: 1.4em 0; }
hr.scene-break::after { content: "* * *"; letter-spacing: 0.4em; }
p.subtitle-tagline { font-style: italic; text-align: center; text-indent: 0;
    margin: 0.2em 0 1em; }
blockquote.scripture { border-left: 3px solid #666; padding: 0.4em 0.8em;
    margin: 1em 0; font-style: italic; }
blockquote.scripture p { text-indent: 0; margin: 0.2em 0; }
blockquote.epigraph { font-style: italic; margin: 1em 2em; text-align: center;
    border-top: 1px solid #999; border-bottom: 1px solid #999; padding: 0.6em 0; }
blockquote.epigraph p { text-indent: 0; margin: 0.2em 0; }
p.italic-quote { font-style: italic; text-align: center; text-indent: 0;
    margin: 1em 2em; }
aside.callout { border-left: 3px solid #666; padding: 0.4em 0.8em; margin: 1em 0; background: #f6f6f6; }
aside.callout p { text-indent: 0; }
table { border-collapse: collapse; margin: 1em auto; }
table th, table td { border: 1px solid #999; padding: 0.3em 0.6em; vertical-align: top; }
ol, ul { margin: 0.6em 0 0.6em 1.6em; }
li { margin-bottom: 0.3em; }
nav#toc ol { list-style: none; padding-left: 1em; }
"""

_CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


@dataclass
class WrittenSection:
    section: Section
    href: str            # relative to OEBPS, e.g. "text/ch-01-foo.xhtml"
    manifest_id: str     # e.g. "sec-3"
    xhtml: str


# --------------------------------------------------------------------------- #
# Inline / block rendering
# --------------------------------------------------------------------------- #

def _render_runs(runs: list[TextRun]) -> str:
    parts: list[str] = []
    for r in runs:
        text = xml_escape(r.text)
        if r.link:
            text = f'<a href={quoteattr(r.link)}>{text}</a>'
        if r.small_caps:
            text = f'<span class="smallcaps">{text}</span>'
        if r.bold and r.italic:
            text = f"<strong><em>{text}</em></strong>"
        elif r.bold:
            text = f"<strong>{text}</strong>"
        elif r.italic:
            text = f"<em>{text}</em>"
        parts.append(text)
    return "".join(parts)


def _render_paragraph(p: Paragraph) -> str:
    classes: list[str] = []
    if p.align == "center":
        classes.append("center")
    if p.drop_cap:
        classes.append("drop-cap")
    attr = f' class="{" ".join(classes)}"' if classes else ""
    return f"<p{attr}>{_render_runs(p.runs)}</p>"


def _render_heading(h: Heading) -> str:
    tag = f"h{h.level}"
    return f"<{tag}>{_render_runs(h.runs)}</{tag}>"


def _render_pullquote(q: PullQuote) -> str:
    runs_html = _render_runs(q.runs)
    if q.role == "subtitle-tagline":
        return f'<p class="subtitle-tagline">{runs_html}</p>'
    if q.role == "plain":
        return f'<p class="italic-quote">{runs_html}</p>'
    inner = f"<p>{runs_html}</p>"
    if q.attribution:
        inner += f'<p class="attribution">— {xml_escape(q.attribution)}</p>'
    return f'<blockquote class="{q.role}">{inner}</blockquote>'


def _render_scene_break(_: SceneBreak) -> str:
    return '<hr class="scene-break"/>'


def _render_list_items(items: list, start: int | None = None) -> str:
    # items is a list[ListItem]; nested children (if any) render as blocks.
    out: list[str] = []
    for it in items:
        inner = _render_runs(it.runs)
        if it.children:
            child_html = "".join(_render_block(b) for b in it.children)
            out.append(f"<li>{inner}{child_html}</li>")
        else:
            out.append(f"<li>{inner}</li>")
    return "".join(out)


def _render_list_block(lb: ListBlock) -> str:
    tag = "ol" if lb.style == "ordered" else "ul"
    starts = lb.item_starts()
    seg_idx_to_start: dict[int, int] = {}
    running = 0
    for i, seg in enumerate(lb.segments):
        if isinstance(seg, ItemSegment):
            seg_idx_to_start[i] = starts[running]
            running += 1

    out: list[str] = []
    for i, seg in enumerate(lb.segments):
        if isinstance(seg, ItemSegment):
            start_val = seg_idx_to_start[i]
            attr = ""
            if lb.style == "ordered" and start_val != 1:
                attr = f' start="{start_val}"'
            out.append(f"<{tag}{attr}>{_render_list_items(seg.items)}</{tag}>")
        elif isinstance(seg, InterruptionSegment):
            out.append("".join(_render_block(b) for b in seg.blocks))
    return "\n".join(out)


def _render_table(t: Table) -> str:
    def cells(row) -> str:
        parts = []
        for c in row.cells:
            attrs = ""
            if c.colspan != 1:
                attrs += f' colspan="{c.colspan}"'
            if c.rowspan != 1:
                attrs += f' rowspan="{c.rowspan}"'
            inner = "".join(_render_block(b) for b in c.blocks)
            parts.append(f"<td{attrs}>{inner}</td>")
        return "".join(parts)

    def hcells(row) -> str:
        parts = []
        for c in row.cells:
            attrs = ""
            if c.colspan != 1:
                attrs += f' colspan="{c.colspan}"'
            if c.rowspan != 1:
                attrs += f' rowspan="{c.rowspan}"'
            inner = "".join(_render_block(b) for b in c.blocks)
            parts.append(f"<th{attrs}>{inner}</th>")
        return "".join(parts)

    parts: list[str] = ["<table>"]
    if t.caption:
        parts.append(f"<caption>{xml_escape(t.caption)}</caption>")
    if t.header_rows:
        parts.append("<thead>")
        for r in t.header_rows:
            parts.append(f"<tr>{hcells(r)}</tr>")
        parts.append("</thead>")
    parts.append("<tbody>")
    for r in t.body_rows:
        parts.append(f"<tr>{cells(r)}</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _render_callout(c: Callout) -> str:
    inner = "".join(_render_block(b) for b in c.content)
    return f'<aside class="callout callout-{c.variant}">{inner}</aside>'


def _render_block(b: Block) -> str:
    if isinstance(b, Paragraph):
        return _render_paragraph(b)
    if isinstance(b, Heading):
        return _render_heading(b)
    if isinstance(b, PullQuote):
        return _render_pullquote(b)
    if isinstance(b, SceneBreak):
        return _render_scene_break(b)
    if isinstance(b, ListBlock):
        return _render_list_block(b)
    if isinstance(b, Table):
        return _render_table(b)
    if isinstance(b, Callout):
        return _render_callout(b)
    raise TypeError(f"unknown block type: {type(b).__name__}")


# --------------------------------------------------------------------------- #
# Section / document assembly
# --------------------------------------------------------------------------- #

_XHTML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}" lang="{lang}">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <link rel="stylesheet" type="text/css" href="../styles.css"/>
</head>
<body>
{body}
</body>
</html>
"""


def _render_section_xhtml(sec: Section, language: str) -> str:
    parts: list[str] = []
    if sec.caption:
        parts.append(f'<p class="chapter-caption">{xml_escape(sec.caption)}</p>')
    parts.append(f"<h1>{xml_escape(sec.title)}</h1>")
    for b in sec.content:
        parts.append(_render_block(b))
    body = "\n".join(parts)
    return _XHTML_TEMPLATE.format(
        lang=xml_escape(language),
        title=xml_escape(sec.title),
        body=body,
    )


def _part_landing_section(part) -> Section:
    """Build a synthetic Section representing a Part's landing page.

    The part marker becomes the caption; the part title becomes the h1.
    Any intro/tagline blocks (e.g. an italic subtitle that follows the
    part-title in the source) are rendered as the body.
    """
    return Section(
        title=part.title,
        kind="chapter",
        slug=part.slug,
        content=list(part.intro),
        caption=part.marker,
    )


def _collect_sections(book: Book) -> list[Section]:
    """Return Section objects to write, in spine order.

    Each Part contributes a synthetic landing section (marker + title +
    optional tagline) immediately before its chapters.
    """
    out: list[Section] = []
    out.extend(book.front_matter)
    for part in book.parts:
        out.append(_part_landing_section(part))
        out.extend(part.chapters)
    out.extend(book.chapters)
    out.extend(book.back_matter)
    return out


def _plan_sections(book: Book, language: str) -> list[WrittenSection]:
    written: list[WrittenSection] = []
    for i, sec in enumerate(_collect_sections(book), start=1):
        href = f"text/{sec.slug}.xhtml"
        mid = f"sec-{i:03d}"
        written.append(WrittenSection(
            section=sec,
            href=href,
            manifest_id=mid,
            xhtml=_render_section_xhtml(sec, language),
        ))
    return written


# --------------------------------------------------------------------------- #
# Nav / OPF / NCX / visible TOC
# --------------------------------------------------------------------------- #

def _render_nav_xhtml(book: Book, written: list[WrittenSection], language: str) -> str:
    by_slug = {w.section.slug: w for w in written}

    items: list[str] = []
    items.append('<h1>Contents</h1>')
    items.append('<nav epub:type="toc" id="toc"><ol>')

    for s in book.front_matter:
        w = by_slug[s.slug]
        items.append(f'  <li><a href={quoteattr(w.href)}>{xml_escape(s.title)}</a></li>')

    for part in book.parts:
        landing = by_slug[part.slug]
        items.append('  <li>')
        items.append(
            f'    <a href={quoteattr(landing.href)}>{xml_escape(part.marker)} — {xml_escape(part.title)}</a>'
        )
        items.append('    <ol>')
        for s in part.chapters:
            w = by_slug[s.slug]
            items.append(f'      <li><a href={quoteattr(w.href)}>{xml_escape(s.title)}</a></li>')
        items.append('    </ol>')
        items.append('  </li>')

    for s in book.chapters:
        w = by_slug[s.slug]
        items.append(f'  <li><a href={quoteattr(w.href)}>{xml_escape(s.title)}</a></li>')

    for s in book.back_matter:
        w = by_slug[s.slug]
        items.append(f'  <li><a href={quoteattr(w.href)}>{xml_escape(s.title)}</a></li>')

    items.append('</ol></nav>')
    body = "\n".join(items)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{xml_escape(language)}" lang="{xml_escape(language)}">
<head>
  <meta charset="utf-8"/>
  <title>Contents</title>
  <link rel="stylesheet" type="text/css" href="styles.css"/>
</head>
<body>
{body}
</body>
</html>
"""


def _render_ncx(book: Book, written: list[WrittenSection], identifier: str) -> str:
    by_slug = {w.section.slug: w for w in written}
    nav_points: list[str] = []
    play_order = 1

    def _emit(title: str, href: str) -> None:
        nonlocal play_order
        nav_points.append(
            f'<navPoint id="np-{play_order}" playOrder="{play_order}">'
            f'<navLabel><text>{xml_escape(title)}</text></navLabel>'
            f'<content src={quoteattr(href)}/></navPoint>'
        )
        play_order += 1

    for s in book.front_matter:
        _emit(s.title, by_slug[s.slug].href)
    for part in book.parts:
        _emit(f"{part.marker} — {part.title}", by_slug[part.slug].href)
        for s in part.chapters:
            _emit(s.title, by_slug[s.slug].href)
    for s in book.chapters:
        _emit(s.title, by_slug[s.slug].href)
    for s in book.back_matter:
        _emit(s.title, by_slug[s.slug].href)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content={quoteattr(identifier)}/>
    <meta name="dtb:depth" content="2"/>
  </head>
  <docTitle><text>{xml_escape(book.metadata.title)}</text></docTitle>
  <navMap>
    {"".join(nav_points)}
  </navMap>
</ncx>
"""


def _render_opf(book: Book, written: list[WrittenSection], identifier: str) -> str:
    md = book.metadata
    manifest_items = [
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '<item id="css" href="styles.css" media-type="text/css"/>',
    ]
    spine_items: list[str] = []
    for w in written:
        manifest_items.append(
            f'<item id={quoteattr(w.manifest_id)} href={quoteattr(w.href)} media-type="application/xhtml+xml"/>'
        )
        spine_items.append(f'<itemref idref={quoteattr(w.manifest_id)}/>')

    manifest = "\n    ".join(manifest_items)
    spine = "\n    ".join(spine_items)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="pub-id" xml:lang="{xml_escape(md.language)}">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="pub-id">{xml_escape(identifier)}</dc:identifier>
    <dc:title>{xml_escape(md.title)}</dc:title>
    <dc:creator>{xml_escape(md.author)}</dc:creator>
    <dc:language>{xml_escape(md.language)}</dc:language>
    <meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>
  </metadata>
  <manifest>
    {manifest}
  </manifest>
  <spine toc="ncx">
    {spine}
  </spine>
</package>
"""


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #

@dataclass
class WrittenEpub:
    output_path: Path
    section_files: list[WrittenSection]
    identifier: str


def write_epub(book: Book, output_path: str | Path) -> WrittenEpub:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    identifier = book.metadata.identifier or f"urn:uuid:{uuid.uuid4()}"
    language = book.metadata.language

    written = _plan_sections(book, language)
    nav_xhtml = _render_nav_xhtml(book, written, language)
    ncx_xml = _render_ncx(book, written, identifier)
    opf_xml = _render_opf(book, written, identifier)

    with zipfile.ZipFile(out, "w") as zf:
        # mimetype must be first, STORED (no compression), no extra fields.
        zi = zipfile.ZipInfo("mimetype")
        zi.compress_type = zipfile.ZIP_STORED
        zf.writestr(zi, "application/epub+zip")

        def _add(name: str, data: str) -> None:
            zf.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)

        _add("META-INF/container.xml", _CONTAINER_XML)
        _add("OEBPS/content.opf", opf_xml)
        _add("OEBPS/nav.xhtml", nav_xhtml)
        _add("OEBPS/toc.ncx", ncx_xml)
        _add("OEBPS/styles.css", _CSS)
        for w in written:
            _add(f"OEBPS/{w.href}", w.xhtml)

    return WrittenEpub(output_path=out, section_files=written, identifier=identifier)
