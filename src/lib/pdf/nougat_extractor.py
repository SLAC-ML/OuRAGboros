import logging
import os
import io

from pathlib import Path
from typing import Optional, Iterable, Tuple, BinaryIO, List
from collections import defaultdict

import pymupdf
import torch

from PIL import Image

from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    AutoProcessor,
    VisionEncoderDecoderModel
)

import lib.config as config
from lib.pdf.extractor import PDFExtractor


# Custom Stopping Criteria
# This is from the Nougat tutorial
class _RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)


class _StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = _RunningVarTorch(norm=True)
        self.varvars = _RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0


class NougatExtractor(PDFExtractor):
    def extract_text(
            self,
            pdf_bytes: BinaryIO,
            filename: str,
            outpath: Optional[Path] = None,
            page_range=None,
            dpi: int = 96,
    ) -> Iterable[Tuple[io.BytesIO, str, List[int]]]:
        """
        :param pdf_bytes: Path to PDF file.
        :param filename: The PDF filename
        :param outpath: The output directory. If None, the text files will not be saved.
        :param dpi: The output DPI. Defaults to 96.
        :param page_range: A list of page numbers to extract. If None, all pages will be
        extracted.
        Defaults to None.
        :return:
        """
        # Step 1: Use Nougat to transform PDF to txt
        # Nougat can preserve math equations and tables

        processor = AutoProcessor.from_pretrained(
            config.pdf_parser_model,
            cache_dir=config.huggingface_model_cache_folder,
        )
        model = VisionEncoderDecoderModel.from_pretrained(
            config.pdf_parser_model,
            cache_dir=config.huggingface_model_cache_folder,
        )

        # use GPU if available
        if torch.cuda.is_available():
            device = "cuda"
            print("GPU available")
        else:
            device = "cpu"
            print("GPU unavailable; using CPU")
        model.to(device)

        # Loop through all pages in the document
        for k, (image, pages) in enumerate(
                self._rasterize_paper(
                    pdf_bytes,
                    filename,
                    outpath=outpath,
                    page_range=page_range,
                    dpi=dpi
                )
        ):
            pil_image = Image.open(image)

            # Preprocess the current image
            pixel_values = processor(images=pil_image,
                                     return_tensors="pt").pixel_values.to(
                device
            )

            # Generate text for the current page
            outputs = model.generate(
                pixel_values,
                min_length=1,
                max_length=3584,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=StoppingCriteriaList([_StoppingCriteriaScores()]),
            )

            # Decode and post-process the generated output for the current page
            generated = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
            generated = processor.post_process_generation(generated, fix_markdown=False)

            page_bytes = io.BytesIO(generated.encode('utf-8'))

            # Save the to text file if desired
            #
            output_file = f"{Path(filename).name}_{pages[k] + 1}.txt"

            if outpath:
                with open(os.path.join(outpath, output_file), "wb") as file:
                    file.write(page_bytes.getbuffer())

            # Append the processed text to the final output with a page separator
            #
            yield page_bytes, output_file, pages

    # PDF to PNG
    #
    def _rasterize_paper(
            self,
            pdf_bytes: BinaryIO,
            filename: str,
            outpath: Optional[Path] = None,
            dpi: int = 96,
            page_range=None,
    ) -> Iterable[io.BytesIO]:
        """
        Rasterize a PDF file to PNG images.

        Args:
            pdf_bytes (io.BytesIO): PDF file bytes.
            outpath (Optional[Path], optional): The output directory. If None, the PIL
            images will not be saved. Defaults to None.
            dpi (int, optional): The output DPI. Defaults to 96.
            page_range (Optional[List[int]], optional): The pages to rasterize. If None,
            all pages will be rasterized. Defaults to None.

        Returns:
            (Iterable[io.BytesIO], List[int]): The PIL images and list of page numbers.
            None.
        """
        try:
            pdf_bytes.seek(0, 0)
            pdf = pymupdf.open(stream=pdf_bytes.read(), filetype="pdf")

            pages = page_range or range(len(pdf))

            for i in pages:
                page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")

                #  Save PNG if desired.
                #
                if outpath:
                    with open(
                            os.path.join(outpath, f"{Path(filename).stem}.{i + 1}.png"),
                            "wb"
                    ) as f:
                        f.write(page_bytes)

                yield io.BytesIO(page_bytes), pages
        except Exception as e:
            logging.critical(e, exc_info=True)
