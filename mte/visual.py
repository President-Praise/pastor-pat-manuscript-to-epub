"""Visual spot-check: render EPUB to PDF, then produce source vs. rendered
contact sheets aligned page-by-page.

Layer 2 in the verify stack. Complements the structural (Layer 1) and raw-text
(Layer 3) diffs by catching typographic regressions those checks are blind to:
headings rendered at body size, missing bold/italic, dropped covers, crushed
line-height, callouts that lost their background.

Not a build gate — this is a batch of side-by-side PNGs a human eyeballs.

Pipeline:
    1. `ebook-convert <in>.epub <out>.pdf --paper-size a5`  (Calibre)
    2. rasterize both PDFs at fixed DPI via PyMuPDF
    3. fingerprint each source page (first ~40 alnum chars, normalized) and
       linear-scan for the matching rendered page; consecutive rendered pages
       up to the next fingerprint constitute the aligned range.
    4. write one `pageNNN.png` per source page: source raster on the left,
       vertically-stacked rendered pages on the right.
"""

from __future__ import annotations

import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

from .verify import _strip_repeating_headers_footers


DEFAULT_DPI = 110
FINGERPRINT_LEN = 40
CALIBRE_PAPER = "a5"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_epub_to_pdf(epub_path: Path, out_pdf: Path,
                       paper: str = CALIBRE_PAPER) -> Path:
    """Shell out to Calibre's ebook-convert. Requires `ebook-convert` on PATH."""
    if shutil.which("ebook-convert") is None:
        raise RuntimeError(
            "ebook-convert not found. Install Calibre "
            "(`brew install --cask calibre`) or add its CLI to PATH."
        )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ebook-convert", str(epub_path), str(out_pdf),
        "--paper-size", paper,
        "--pdf-page-numbers",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_pdf


def rasterize_pdf(pdf_path: Path, dpi: int = DEFAULT_DPI) -> list[Image.Image]:
    imgs: list[Image.Image] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pm = page.get_pixmap(dpi=dpi)
            imgs.append(Image.open(BytesIO(pm.tobytes("png"))).convert("RGB"))
    return imgs


def extract_page_texts(pdf_path: Path) -> list[str]:
    out: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            out.append(page.get_text("text"))
    return out


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #

def _fingerprint(text: str, length: int = FINGERPRINT_LEN,
                 skip_prefix: int = 0) -> str:
    """Alnum-only, casefolded, NFKC-normalized prefix. skip_prefix drops the
    leading N alnum chars — useful when a rendered page starts with a page
    number that pollutes the fingerprint of the FIRST source page it maps to
    (source page has no leading number in body flow)."""
    norm = unicodedata.normalize("NFKC", text)
    alnum = "".join(c.casefold() for c in norm if c.isalnum())
    return alnum[skip_prefix:skip_prefix + length]


def _find_fingerprint(fp: str, haystack_norm: list[str],
                      start: int = 0) -> int:
    if not fp:
        return -1
    for i in range(start, len(haystack_norm)):
        if fp in haystack_norm[i]:
            return i
    return -1


@dataclass
class PageAlignment:
    src_page: int              # 1-based source PDF page
    rendered_pages: list[int]  # 1-based rendered PDF pages, [] if unmatched
    note: str = ""
    blank: bool = False        # source page had no substantive text after header/footer strip


def _alnum_norm(text: str) -> str:
    return "".join(c.casefold()
                   for c in unicodedata.normalize("NFKC", text)
                   if c.isalnum())


def _candidate_fingerprints(text: str, length: int = FINGERPRINT_LEN,
                            max_starts: int = 8,
                            short_len: int = 18) -> list[str]:
    """Yield fingerprints from several offsets into the page's alnum stream.

    Handles two failure modes:
      - short pages (part banners with ~50 alnum chars): the single 40-char
        fingerprint may include leftover header/footer noise; we fall back to
        a shorter `short_len` window that skips the noisy first chars.
      - normal pages whose leading text is dominated by a heading; later
        offsets then hit body content that survives the reflow.
    """
    alnum = _alnum_norm(text)
    if not alnum:
        return []
    out: list[str] = []
    if len(alnum) >= length:
        step = max(1, (len(alnum) - length) // max(1, max_starts - 1))
        seen: set[str] = set()
        for k in range(max_starts):
            start = k * step
            if start + length > len(alnum):
                break
            fp = alnum[start:start + length]
            if fp not in seen:
                out.append(fp)
                seen.add(fp)
    # Short-window fallbacks — several windows over the whole alnum stream.
    if len(alnum) >= short_len:
        step = max(1, (len(alnum) - short_len) // 4)
        for k in range(5):
            start = k * step
            if start + short_len > len(alnum):
                break
            fp = alnum[start:start + short_len]
            if fp not in out:
                out.append(fp)
    else:
        out.append(alnum)
    return out


def align_pages(src_texts: list[str],
                rendered_texts: list[str]) -> list[PageAlignment]:
    """Best-effort alignment. Missing source pages (front-matter, TOC) end up
    with rendered_pages=[] and are skipped in the contact sheet."""
    # Strip repeating headers/footers on both sides so fingerprints aren't
    # dominated by "Pastor PAT Akem-Vingir / Book Title" running headers
    # (source) or Calibre's injected "N" page-number (rendered).
    src_clean = _strip_repeating_headers_footers(src_texts)
    rendered_clean = _strip_repeating_headers_footers(rendered_texts)
    rendered_norm = [_alnum_norm(t) for t in rendered_clean]

    alignments: list[PageAlignment] = []
    cursor = 0
    BLANK_THRESHOLD = 25  # alnum chars — pages below this are structural spacers
    for si, src in enumerate(src_clean, start=1):
        alnum_len = len(_alnum_norm(src))
        candidates = _candidate_fingerprints(src)
        if not candidates or alnum_len < BLANK_THRESHOLD:
            alignments.append(PageAlignment(
                si, [], "blank/spacer page", blank=True))
            continue
        hit = -1
        for fp in candidates:
            hit = _find_fingerprint(fp, rendered_norm, start=cursor)
            if hit < 0:
                hit = _find_fingerprint(fp, rendered_norm, start=0)
            if hit >= 0:
                break
        if hit < 0:
            alignments.append(PageAlignment(si, [], "no fingerprint match"))
            continue
        alignments.append(PageAlignment(si, [hit + 1]))
        cursor = hit  # allow next source page to reuse this render page

    # Second pass: extend each alignment's rendered range to include pages
    # up to (but not including) the next matched alignment's first page.
    matched = [a for a in alignments if a.rendered_pages]
    for k, a in enumerate(matched):
        first = a.rendered_pages[0]
        next_first = (matched[k + 1].rendered_pages[0]
                      if k + 1 < len(matched) else len(rendered_texts) + 1)
        # Guard against out-of-order matches (fingerprint collisions): if the
        # next start is <= current start, treat this source page as covering
        # only its own matched page.
        if next_first > first:
            a.rendered_pages = list(range(first, next_first))
    return alignments


# --------------------------------------------------------------------------- #
# Contact sheets
# --------------------------------------------------------------------------- #

def _label_bar(width: int, text: str, height: int = 22) -> Image.Image:
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 3), text, fill=(230, 230, 230), font=font)
    return img


def _stack_vertical(imgs: list[Image.Image], width: int,
                    gap: int = 8) -> Image.Image:
    scaled = []
    total_h = 0
    for im in imgs:
        ratio = width / im.width
        new = im.resize((width, int(im.height * ratio)), Image.LANCZOS)
        scaled.append(new)
        total_h += new.height
    total_h += gap * max(0, len(scaled) - 1)
    if total_h == 0:
        return Image.new("RGB", (width, 1), (255, 255, 255))
    canvas = Image.new("RGB", (width, total_h), (255, 255, 255))
    y = 0
    for im in scaled:
        canvas.paste(im, (0, y))
        y += im.height + gap
    return canvas


def make_contact_sheet(src_img: Image.Image,
                       rendered_imgs: list[Image.Image],
                       label: str,
                       column_w: int = 560) -> Image.Image:
    src_ratio = column_w / src_img.width
    src_resized = src_img.resize(
        (column_w, int(src_img.height * src_ratio)), Image.LANCZOS)
    if rendered_imgs:
        right = _stack_vertical(rendered_imgs, column_w)
    else:
        right = Image.new("RGB", (column_w, src_resized.height),
                          (245, 220, 220))
        d = ImageDraw.Draw(right)
        d.text((10, 10), "(no rendered match)", fill=(120, 30, 30))
    left_bar = _label_bar(column_w, f"SOURCE  {label.split('  |  ')[0]}")
    right_bar = _label_bar(column_w, f"RENDERED  {label.split('  |  ')[1]}"
                           if "  |  " in label else "RENDERED")
    left = Image.new("RGB", (column_w, left_bar.height + src_resized.height),
                     (255, 255, 255))
    left.paste(left_bar, (0, 0))
    left.paste(src_resized, (0, left_bar.height))
    right_full = Image.new(
        "RGB", (column_w, right_bar.height + right.height), (255, 255, 255))
    right_full.paste(right_bar, (0, 0))
    right_full.paste(right, (0, right_bar.height))
    h = max(left.height, right_full.height)
    canvas = Image.new("RGB", (column_w * 2 + 12, h), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right_full, (column_w + 12, 0))
    return canvas


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

@dataclass
class VisualReport:
    src_pages: int
    rendered_pages: int
    aligned: int          # source pages that matched at least one rendered page
    unmatched: list[int]  # source page numbers with real content but no match
    blank: list[int]      # source page numbers that were structural spacers
    out_dir: Path

    def format(self) -> str:
        substantive = self.src_pages - len(self.blank)
        pct = 100.0 * self.aligned / max(1, substantive)
        lines = [
            f"source pages: {self.src_pages} ({len(self.blank)} blank spacers)",
            f"rendered pages: {self.rendered_pages}",
            f"aligned: {self.aligned}/{substantive} substantive ({pct:.1f}%)",
        ]
        if self.unmatched:
            preview = ", ".join(str(p) for p in self.unmatched[:20])
            more = f" (+{len(self.unmatched) - 20} more)" if len(self.unmatched) > 20 else ""
            lines.append(f"unmatched non-blank source pages: {preview}{more}")
        lines.append(f"contact sheets: {self.out_dir}")
        return "\n".join(lines)


def verify_visual(pdf_path: Path, epub_path: Path, out_dir: Path,
                  dpi: int = DEFAULT_DPI,
                  paper: str = CALIBRE_PAPER,
                  keep_rendered_pdf: bool = True) -> VisualReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_pdf = out_dir / f"{epub_path.stem}-rendered.pdf"
    render_epub_to_pdf(epub_path, rendered_pdf, paper=paper)

    src_texts = extract_page_texts(pdf_path)
    rendered_texts = extract_page_texts(rendered_pdf)
    alignments = align_pages(src_texts, rendered_texts)

    src_imgs = rasterize_pdf(pdf_path, dpi=dpi)
    rendered_imgs = rasterize_pdf(rendered_pdf, dpi=dpi)

    aligned_count = 0
    unmatched: list[int] = []
    blank: list[int] = []
    for a in alignments:
        if not a.rendered_pages:
            if a.blank:
                blank.append(a.src_page)
            else:
                unmatched.append(a.src_page)
            continue
        aligned_count += 1
        rendered_slice = [rendered_imgs[i - 1] for i in a.rendered_pages
                          if 1 <= i <= len(rendered_imgs)]
        label = (f"p{a.src_page:03d}  |  "
                 f"rendered p{a.rendered_pages[0]}"
                 f"{'-' + str(a.rendered_pages[-1]) if len(a.rendered_pages) > 1 else ''}")
        sheet = make_contact_sheet(src_imgs[a.src_page - 1],
                                   rendered_slice, label)
        sheet.save(out_dir / f"page{a.src_page:03d}.png", "PNG", optimize=True)

    if not keep_rendered_pdf:
        rendered_pdf.unlink(missing_ok=True)

    return VisualReport(
        src_pages=len(src_texts),
        rendered_pages=len(rendered_texts),
        aligned=aligned_count,
        unmatched=unmatched,
        blank=blank,
        out_dir=out_dir,
    )
