import fitz  # PyMuPDF
import pytesseract
import json
import os
from PIL import Image
from tqdm import tqdm

PDF_PATH = r"C:\Users\zbook\OneDrive\Desktop\Language Assistant\data\raw_pdf\OxfordDictionary.pdf"
OUTPUT_PATH = r"C:\Users\zbook\OneDrive\Desktop\Language Assistant\data\processed\DictionaryToText.json"

def extract_text_with_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page_number in tqdm(range(len(doc)), desc="OCR Processing"):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # OCR with Tesseract
        text = pytesseract.image_to_string(image)
        all_text.append({
            "page": page_number + 1,
            "text": text
        })

    return all_text
if __name__ == "__main__":
    print("🔍 Running OCR...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    ocr_data = extract_text_with_pymupdf(PDF_PATH)

    print(f"\n📄 Saving to: {os.path.abspath(OUTPUT_PATH)}")
    print(f"📄 Total pages extracted: {len(ocr_data)}")

    # Preview the first page
    if ocr_data and ocr_data[0]['text'].strip():
        print(f"\n📄 Sample text from Page 1:\n{ocr_data[0]['text'][:500]}")
    else:
        print("⚠️ No text extracted from page 1. Check if OCR is working properly.")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ OCR complete. Text saved to: {OUTPUT_PATH}")
