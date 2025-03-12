from pathlib import Path
from typing import Optional, Iterable, Tuple, BinaryIO, List

import io


class PDFExtractor:
    """
    This is the base class for the PDF extractor. Other implementations can subclass this
    so as to support multiple PDF parsing implementations (e.g., cloud-based OCR, etc.).
    """

    def extract_text(
            self,
            pdf_bytes: BinaryIO,
            filename: str,
            outpath: Optional[Path] = None,
            page_range: Optional[List[int]] = None
    ) -> Iterable[Tuple[io.BytesIO, str, List[int]]]:
        raise NotImplementedError()
