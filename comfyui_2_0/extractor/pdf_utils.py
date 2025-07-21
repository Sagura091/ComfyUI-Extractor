import pdfplumber, pytesseract, io, PIL.Image as Image, requests
from pathlib import Path
from .caption_api import auto_caption
from .settings import OUT_DIR

def crop_save(page, img_dict, out_dir, pnum, idx):
    bbox = (img_dict["x0"], img_dict["top"], img_dict["x1"], img_dict["bottom"])
    img = page.crop(bbox).to_image(resolution=300).original
    path = out_dir / f"p{pnum}_{idx}.png"; img.save(path, "PNG")
    return path

def nearby_caption(page, img_dict, max_px=60):
    y_bottom = img_dict["bottom"]
    words = page.extract_words()
    line = [w for w in words if 0 < w["top"] - y_bottom < max_px]
    if not line: return ""
    line = sorted(line, key=lambda w: w["x0"])
    return " ".join(w["text"] for w in line)

def ocr_caption(path):
    return pytesseract.image_to_string(Image.open(path))[:120]

def process_pdf(path: Path, push_record):
    doc_id = path.stem
    with pdfplumber.open(path) as pdf:
        for pnum, page in enumerate(pdf.pages, 1):
            for idx, img in enumerate(page.images):
                out_dir = (OUT_DIR / doc_id); out_dir.mkdir(parents=True, exist_ok=True)
                p = crop_save(page, img, out_dir, pnum, idx)
                cap = nearby_caption(page, img) or auto_caption(p) or ocr_caption(p)
                push_record(doc_id, f"p{pnum}_{idx}", cap, p,
                            raw_text=page.extract_text() or "")
