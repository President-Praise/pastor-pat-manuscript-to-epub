# Manuscript to EPUB

A Python toolchain that converts raw PDF manuscripts into professionally formatted EPUB and PDF e-books.

Built for a publishing project involving 6+ book titles.

## What it does

1. **Extracts** text and structure from manuscript PDFs
2. **Classifies** content blocks (headings, body, pull quotes, sidebars)
3. **Builds** an intermediate representation of the book structure
4. **Generates** clean EPUB files with proper chapter breaks, formatting, and cover art
5. **Verifies** output quality with automated checks

## Architecture

```
manuscript_to_epub.py          # main entry point
├── mte/
│   ├── pdf_extract.py         # PDF text extraction
│   ├── preprocess.py          # text cleanup
│   ├── classify.py            # content block classification
│   ├── build_ir.py            # intermediate representation
│   ├── ir.py                  # IR data structures
│   ├── epub_writer.py         # EPUB generation
│   ├── verify.py              # output verification
│   ├── visual.py              # visual inspection tools
│   └── tiers.py               # quality tier definitions
├── tools/
│   └── survey_pdfs.py         # batch PDF analysis
├── 001/                       # source manuscripts (PDF)
├── Covers/                    # book cover assets
└── out/                       # generated EPUBs
```

## Books processed

- Anagkazō
- Awake!
- Occupy Till I Come
- Ancient Wells Opening
- Believers as Priests and Kings
- Audacious Prayers

## Setup

```bash
cp .env.example .env
pip install -r requirements.txt   # if requirements exist
python manuscript_to_epub.py
```
