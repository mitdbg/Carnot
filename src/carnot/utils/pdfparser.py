import io
from pypdf import PdfReader


def get_text_from_pdf(filename, pdf_bytes, pdfprocessor="pypdf", enable_file_cache=True, file_cache_dir="/tmp"):
    """
    Extract text from a PDF file using pypdf.
    
    Args:
        filename: The name of the PDF file
        pdf_bytes: The PDF file as bytes
        pdfprocessor: PDF processor to use (only "pypdf" supported)
        enable_file_cache: Whether to use file caching (ignored for pypdf)
        file_cache_dir: Directory for file caching (ignored for pypdf)
    
    Returns:
        str: The extracted text from the PDF
    """
    if pdfprocessor == "pypdf":
        pdf = PdfReader(io.BytesIO(pdf_bytes))
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"
        return all_text
    else:
        raise ValueError(f"Unsupported PDF processor: {pdfprocessor}. Only 'pypdf' is supported.")