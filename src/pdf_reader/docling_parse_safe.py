"""
docling_parse_safe.py - a Docling PDF backend for the pages that crash docling_parse's renderer.

docling_parse 7.22 (Docling's default PDF backend) renders page images on its worker threads, and on pages with a
very large vector drawing - MTL-LoRA's page 1: a t-SNE scatter plot of 4,140 paths - that renderer corrupts the
heap and kills the process (Windows 0xc0000374) in about a quarter to half of the first conversions. Measured in
fresh processes (bench/crash_first_page.py, scratch tests in docs/JOURNEY.md section 26):
  that page, default backend                                         2-8 crashed of 8-14
  other pages and papers (up to 616 paths per page), default backend  0 of 8 each, never in any other run
  that page, docling_parse text + pdfium page images (this backend)   0 of 20
This backend keeps docling_parse for the text and draws the page images with pdfium, as Docling's pdfium backend
does (1.5x, then resized), under Docling's pdfium lock. pdfium's pixels differ a little, which can move the
layout and table models, so it is used only for pages with more than HEAVY_PATHS vector paths (1 page of 298 in
seven papers); every other page is converted exactly as before (stream.Engine.index_page chooses per page).
"""
from io import BytesIO
from pathlib import Path
from typing import Iterator, Optional

import pypdfium2 as pdfium
from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend, ThreadedDoclingParsePageBackend
from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.utils.locks import pypdfium2_lock
from docling_core.types.doc import BoundingBox, CoordOrigin
from PIL import Image

HEAVY_PATHS = 1500  # pages above this many vector paths use this backend (the busiest safe page had 616)


def backend_options():
    return ThreadedDoclingParseBackendOptions(parser_threads=1, render_pages=False)


class PdfiumImagePageBackend(ThreadedDoclingParsePageBackend):
    def __init__(self, result, pdf):
        super().__init__(result, rendered=False)
        self._pdf = pdf

    def get_page_image(self, scale: float = 1, cropbox: Optional[BoundingBox] = None) -> Image.Image:
        """As PyPdfiumPageBackend.get_page_image (docling.backend.pypdfium2_backend)."""
        with pypdfium2_lock:
            page = self._pdf[self.page_no - 1]
            try:
                width, height = page.get_width(), page.get_height()
                if not cropbox:
                    cropbox = BoundingBox(l=0, r=width, t=0, b=height, coord_origin=CoordOrigin.TOPLEFT)
                    padbox = BoundingBox(l=0, r=0, t=0, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)
                else:
                    padbox = cropbox.to_bottom_left_origin(height).model_copy()
                    padbox.r = width - padbox.r
                    padbox.t = height - padbox.t
                bitmap = page.render(scale=scale * 1.5, rotation=0, crop=padbox.as_tuple())
                image = bitmap.to_pil().copy()
                bitmap.close()
            finally:
                page.close()
        return image.resize(size=(round(cropbox.width * scale), round(cropbox.height * scale)))


class PdfiumImageDocumentBackend(ThreadedDoclingParseDocumentBackend):
    """docling_parse for the text (no rendering), pdfium for the page images."""

    def __init__(self, in_doc, path_or_stream, options=None):
        super().__init__(in_doc, path_or_stream, options or backend_options())
        src = path_or_stream.getvalue() if isinstance(path_or_stream, BytesIO) else str(Path(path_or_stream))
        with pypdfium2_lock:
            self._pdf = pdfium.PdfDocument(src)

    def iter_pages(self) -> Iterator[ThreadedDoclingParsePageBackend]:
        self._iterating = True
        for result in self.parser.iterate_results():
            yield PdfiumImagePageBackend(result, self._pdf)
        self._iterating = False

    def unload(self) -> None:
        super().unload()
        with pypdfium2_lock:
            if self._pdf is not None:
                self._pdf.close()
                self._pdf = None
