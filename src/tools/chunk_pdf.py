"""
This file creates the `chunk_pdf` uv tool (https://docs.astral.sh/uv/concepts/tools/),
which is used to parse an academic paper PDF from the command line via Nougat
(https://github.com/facebookresearch/nougat).

Run `uv run chunk_pdf` on the command line for usage instructions.
"""
from pathlib import Path
import argparse

from lib.pdf.nougat_extractor import NougatExtractor
from tools.constants import FILENAME_ARG, OUTPATH_ARG

parser = argparse.ArgumentParser(
    prog="uv run chunk_pdf", description="Chunks a PDF into a bunch of text files.",
)

parser.add_argument(FILENAME_ARG, help="Input PDF file.")
parser.add_argument(
    OUTPATH_ARG, help="Output directory to store processed text files.", default="."
)


def main():
    args = parser.parse_args()
    pdf_extractor = NougatExtractor()
    with open(args.filename, "rb") as file:
        for k, (text_bytes, text_file_name, pages) in enumerate(
            pdf_extractor.extract_text(
                pdf_bytes=file,
                filename=Path(args.filename).name,
                outpath=Path(args.outpath),
                page_range=None,
            )
        ):
            print(f"Processing page {pages[k] + 1} [{k + 1}/{len(pages)}]...")


if __name__ == "__main__":
    main()
