"""
docx_to_sentences.py
────────────────────
Extracts every sentence from one or more Word (.docx) files and writes
each sentence as its own row in a CSV — ready for LLM training / sparse-
matrix construction.

Usage
─────
  # Single file
  python docx_to_sentences.py essay.docx

  # Multiple files
  python docx_to_sentences.py *.docx

  # Whole folder (recursive)
  python docx_to_sentences.py --dir ~/Documents/Writing

  # Custom output path
  python docx_to_sentences.py --dir ~/Writing --out corpus.csv

  # Skip short fragments (default: 10 chars)
  python docx_to_sentences.py --dir ~/Writing --min-len 20

Dependencies
────────────
  pip install python-docx
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

try:
    from docx import Document
except ImportError:
    sys.exit("Missing dependency. Run:  pip install python-docx")


# ─── Sentence Splitter ────────────────────────────────────────────────────────

# Titles / abbreviations that should NOT trigger a sentence break
ABBREVS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "rev", "gen", "sgt",
    "cpl", "pvt", "rep", "sen", "gov", "lt", "col", "maj", "brig",
    "st", "ave", "blvd", "dept", "approx", "est", "orig", "incl",
    "etc", "vs", "fig", "vol", "no", "pp", "ed", "trans", "jan",
    "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "u.s", "u.k", "e.g", "i.e", "a.m", "p.m",
}

# Compiled once: matches sentence-ending punctuation followed by whitespace +
# an uppercase letter (or end of string).
_SENT_END = re.compile(
    r"""
    (?<!\w\.\w.)        # not inside an acronym like U.S.A.
    (?<![A-Z][a-z]\.)   # not after a single-cap abbrev like Mr.
    (?<=\.|\?|!|…)      # must follow sentence-ending punctuation
    (?:\s*[""'»])?      # optional closing quote
    \s+                 # whitespace between sentences
    (?=[A-Z"'«\(])      # next token starts uppercase or opening quote
    """,
    re.VERBOSE,
)


def split_sentences(text: str) -> list[str]:
    """
    Regex-based sentence splitter robust enough for typical prose.
    Handles:
      • Common abbreviations (Mr., Dr., Jan., etc.)
      • Ellipses  (… or ...)
      • Closing quotes  ("She said." He replied.)
      • Parenthetical openings
      • Numeric decimals  (3.14)
      • All-caps abbreviations  (NASA. The agency...)
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Protect known abbreviations by replacing their periods temporarily
    def protect_abbrevs(m):
        word = m.group(1).lower()
        if word in ABBREVS:
            return m.group(0).replace(".", "·")   # ← placeholder
        return m.group(0)

    text = re.sub(r"\b([A-Za-z]{1,6})\.", protect_abbrevs, text)

    # Protect decimal numbers  (3.14 → 3·14)
    text = re.sub(r"(\d)\.(\d)", r"\1·\2", text)

    # Protect ellipses  (... → ···)
    text = re.sub(r"\.{2,}", lambda m: "·" * len(m.group()), text)

    # Protect single-capital initials  (J. Smith → J· Smith)
    text = re.sub(r"\b([A-Z])\.", r"\1·", text)

    # Split on sentence boundaries
    raw = _SENT_END.split(text)

    # Restore placeholders
    sentences = [s.replace("·", ".").strip() for s in raw if s.strip()]
    return sentences


# ─── DOCX Reader ──────────────────────────────────────────────────────────────

def iter_paragraphs(docx_path: Path):
    """
    Yields (paragraph_index, paragraph_text) from a .docx file,
    skipping empty paragraphs.
    """
    doc = Document(str(docx_path))
    for idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            yield idx, text


# ─── Main Processing ──────────────────────────────────────────────────────────

CSV_HEADER = [
    "sentence_id",   # global sequential ID across all files
    "source_file",   # original filename (no path, for portability)
    "paragraph_idx", # which paragraph in that file
    "sentence_idx",  # which sentence within that paragraph
    "sentence",      # the actual text
    "char_count",    # quick quality signal
    "word_count",    #   "        "
]


def process_files(docx_paths: list[Path], out_csv: Path, min_len: int = 10):
    total_sentences = 0
    skipped = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(CSV_HEADER)

        for docx_path in docx_paths:
            print(f"  Processing: {docx_path.name}")
            try:
                for para_idx, para_text in iter_paragraphs(docx_path):
                    sentences = split_sentences(para_text)
                    for sent_idx, sentence in enumerate(sentences):
                        if len(sentence) < min_len:
                            skipped += 1
                            continue
                        writer.writerow([
                            total_sentences + 1,       # sentence_id (1-based)
                            docx_path.name,            # source_file
                            para_idx,                  # paragraph_idx
                            sent_idx,                  # sentence_idx
                            sentence,                  # sentence
                            len(sentence),             # char_count
                            len(sentence.split()),     # word_count
                        ])
                        total_sentences += 1
            except Exception as e:
                print(f"  ⚠  Could not read {docx_path.name}: {e}", file=sys.stderr)

    print(f"\n✓ Done.")
    print(f"  Sentences written : {total_sentences}")
    print(f"  Fragments skipped : {skipped}  (< {min_len} chars)")
    print(f"  Output            : {out_csv}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def collect_docx(args) -> list[Path]:
    paths = []

    # --dir mode: walk directory recursively
    if args.dir:
        root = Path(args.dir).expanduser()
        if not root.is_dir():
            sys.exit(f"Directory not found: {root}")
        paths = sorted(root.rglob("*.docx"))
        if not paths:
            sys.exit(f"No .docx files found under {root}")

    # Positional file arguments
    elif args.files:
        for f in args.files:
            p = Path(f).expanduser()
            if not p.exists():
                print(f"  ⚠  File not found, skipping: {p}", file=sys.stderr)
            elif p.suffix.lower() != ".docx":
                print(f"  ⚠  Not a .docx file, skipping: {p}", file=sys.stderr)
            else:
                paths.append(p)
        if not paths:
            sys.exit("No valid .docx files to process.")
    else:
        sys.exit("Provide file paths or use --dir. Run with --help for usage.")

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract sentences from .docx files into a CSV for LLM training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="*", help="One or more .docx files")
    parser.add_argument("--dir", metavar="PATH",
                        help="Recursively process all .docx files in this directory")
    parser.add_argument("--out", metavar="FILE", default="sentences.csv",
                        help="Output CSV path (default: sentences.csv)")
    parser.add_argument("--min-len", type=int, default=10, metavar="N",
                        help="Minimum sentence length in characters (default: 10)")
    args = parser.parse_args()

    docx_paths = collect_docx(args)
    out_csv = Path(args.out).expanduser()

    print(f"\nFound {len(docx_paths)} .docx file(s) to process.")
    print(f"Output → {out_csv}\n")

    process_files(docx_paths, out_csv, min_len=args.min_len)


if __name__ == "__main__":
    main()
