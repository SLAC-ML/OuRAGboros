from pathlib import Path
import argparse

from lib.chunk import extract_pdf_text

parser = argparse.ArgumentParser(
    prog='Chunk PDF',
    description='Chunks a PDF into a bunch of text files.',
)

parser.add_argument('filename')
parser.add_argument('outpath')

def main():
    args = parser.parse_args()
    for (text, text_file) in extract_pdf_text(
        pdf_path=Path(args.filename),
        outpath=Path(args.outpath),
    ):
        pass

if __name__ == '__main__':
    main()
