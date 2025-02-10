from pathlib import Path
import argparse

from lib.pdf import extract_text

parser = argparse.ArgumentParser(
    prog='Chunk PDF',
    description='Chunks a PDF into a bunch of text files.',
)

parser.add_argument('filename')
parser.add_argument('outpath')


def main():
    args = parser.parse_args()

    with open(args.filename, 'rb') as file:
        for k, (text_bytes, text_file_name, pages) in enumerate(extract_text(
                pdf_bytes=file,
                filename=Path(args.filename).name,
                outpath=Path(args.outpath),
                page_range=[0, 7, 9]
        )):
            print(f"Processing page {pages[k] + 1} [{k + 1}/{len(pages)}]...")


if __name__ == '__main__':
    main()
