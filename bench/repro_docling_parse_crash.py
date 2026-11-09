import sys
from docling.document_converter import DocumentConverter
res = DocumentConverter().convert(sys.argv[1], page_range=(1, 1))
print("ok", len(res.document.texts), flush=True)
