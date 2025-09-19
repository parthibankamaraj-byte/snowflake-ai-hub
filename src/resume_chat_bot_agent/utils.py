from __future__ import annotations

from typing import List

import io

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore
try:
    import pytesseract
    from PIL import Image
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore
    Image = None  # type: ignore
try:
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover
    pdfium = None  # type: ignore
try:
    import pypdf
except Exception:  # pragma: no cover
    pypdf = None  # type: ignore


def extract_text_from_pdf(content_bytes: bytes) -> str:
    # Prefer pdfplumber for better extraction fidelity
    if pdfplumber is not None:
        try:
            texts: List[str] = []
            with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
                for page in pdf.pages:
                    texts.append(page.extract_text() or "")
            text = "\n".join(texts)
            if text.strip():
                return text
        except Exception:
            pass
    # Fallback to pypdf
    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            texts2: List[str] = []
            for page in reader.pages:
                try:
                    texts2.append(page.extract_text() or "")
                except Exception:
                    continue
            text2 = "\n".join(texts2)
            if text2.strip():
                return text2
        except Exception:
            pass

    # OCR fallback for scanned PDFs
    if pdfium is not None and pytesseract is not None and Image is not None:
        try:
            pdf = pdfium.PdfDocument(io.BytesIO(content_bytes))
            pages_text: List[str] = []
            for i in range(len(pdf)):
                page = pdf[i]
                pil_image = page.render(scale=2).to_pil()
                txt = pytesseract.image_to_string(pil_image)
                pages_text.append(txt or "")
            return "\n".join(pages_text)
        except Exception:
            return ""
    return ""


def extract_text_from_txt(content_bytes: bytes) -> str:
    try:
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return content_bytes.decode("latin-1", errors="ignore")


