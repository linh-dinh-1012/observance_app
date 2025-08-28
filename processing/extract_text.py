import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

def extract_text_with_hybrid_mode(pdf_path, cutoff_ratio=0.85):
    output = []
    log = []

    with pdfplumber.open(pdf_path) as pdf:
        images = convert_from_path(pdf_path, dpi=300)

        for i, page in enumerate(pdf.pages):
            label = f"--- Page {i+1} ---"
            height = page.height

            # Top of page area (cutoff 85%)
            top = page.within_bbox((0, 0, page.width, height * cutoff_ratio))
            pdf_text = top.extract_text() or ""

            # OCR from image
            ocr_text = pytesseract.image_to_string(images[i], lang='fra')

            # Compare length to select result
            if len(ocr_text.strip()) > len(pdf_text.strip()):
                final_text = ocr_text
                log.append(f"Page {i+1}: OCR (longer)")
                output.append(f"{label} (OCR)\n {final_text}")
            else:
                final_text = pdf_text
                log.append(f"Page {i+1}: Text OK")
                output.append(f"{label}\n {final_text}")
    
    return "\n\n".join(output), log