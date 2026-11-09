import sys
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption

conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(backend=PyPdfiumDocumentBackend)})
res = conv.convert(sys.argv[1], page_range=(1, 1))
print("ok", len(res.document.texts), flush=True)
