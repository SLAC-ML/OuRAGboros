import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoProcessor, VisionEncoderDecoderModel
import torch

from typing import Optional, List
import io
from pathlib import Path

from PIL import Image

from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict

import re

# Step 1: Use Nougat to transform PDF to txt
# Nougat can perserve math equations and tables

processor = AutoProcessor.from_pretrained("facebook/nougat-small")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")

# use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

filepath = "/Users/zoeee_ji/Desktop/pdg_short.pdf"

# PDF to PNG
def rasterize_paper(
    pdf: Path,
    outpath: Optional[Path] = None,
    dpi: int = 96,
    return_pil=False,
    pages=None,
) -> Optional[List[io.BytesIO]]:
    """
    Rasterize a PDF file to PNG images.

    Args:
        pdf (Path): The path to the PDF file.
        outpath (Optional[Path], optional): The output directory. If None, the PIL images will be returned instead. Defaults to None.
        dpi (int, optional): The output DPI. Defaults to 96.
        return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.
        pages (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

    Returns:
        Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
    """

    pillow_images = []
    if outpath is None:
        return_pil = True
    try:
        if isinstance(pdf, (str, Path)):
            pdf = fitz.open(pdf)
        if pages is None:
            pages = range(len(pdf))
        for i in pages:
            page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
            if return_pil:
                pillow_images.append(io.BytesIO(page_bytes))
            else:
                with (outpath / ("%02d.png" % (i + 1))).open("wb") as f:
                    f.write(page_bytes)
    except Exception:
        pass
    if return_pil:
        return pillow_images
    

images = rasterize_paper(pdf=filepath, return_pil=True)

# Custon Stopping Citeria
# This is from the Nougat tutorial
class RunningVarTorch:
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


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
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


# Generate txt file from images through Nougat
pdg_text = ""

# Loop through all pages in the document
for page_num in range(len(images)):
    print(f"Processing page {page_num + 1}...")

    image = Image.open(images[page_num])

    # Preprocess the current image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate text for the current page
    outputs = model.generate(
        pixel_values,
        min_length=1,
        max_length=3584,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
    )

    # Decode and post-process the generated output for the current page
    generated = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
    generated = processor.post_process_generation(generated, fix_markdown=False)

    # Append the processed text to the final output with a page separator
    pdg_text += f"### Page {page_num + 1} ###\n{generated}\n\n"

# Save the final concatenated output to a text file
output_file = "pdg_text.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(pdg_text)

print(f"All pages processed and saved to {output_file}")



# Step 2: Chunk the outputed txt file through LangChain

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000000,  # Max characters per chunk
    chunk_overlap=0,  # No overlap between chunks
    separators=[r"\n### Page \d+ ###"]  # Split by "### Page X ###" headers (regex)
)

# Split the document by pages
chunks = text_splitter.split_text(pdg_text)

# Convert chunks into LangChain Document objects
documents = [Document(page_content=chunk) for chunk in chunks]

# Print the resulting chunks (pages)
for i, chunk in enumerate(chunks):
    print(f"\n{chunk}\n")


# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,  # Number of characters per chunk
#     chunk_overlap=200,  # Number of overlapping characters between chunks
#     separators=["\n\n", "\n", " ", ""]
# )

# # Split the extracted PDF text into chunks
# chunks = text_splitter.split_text(pdg_text)

# # Convert chunks into LangChain Document objects
# documents = [Document(page_content=chunk) for chunk in chunks]





# Print first few chunks
# for i, doc in enumerate(documents[:2]):  # Display first 5 chunks
#     print(f"Chunk {i+1}:")
#     print(doc.page_content)
#     print("-" * 40)