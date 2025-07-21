"""
FastAPI micro‑service that ingests single files **or whole batches**,
extracts every bitmap + best caption, embeds text into Chroma, and
drops PNG + JSON side‑cars into /mnt/extract.

Supported extensions (see process_single):
    • .pdf   • .pptx   • .xlsx / .xlsm   • .md
"""
import os, json, traceback, tarfile, zipfile, shutil, uuid, pathlib
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import HttpClient
from .settings import (
    RAW_DIR, OUT_DIR,
    CHROMA_HOST, CHROMA_PORT,
    EMBED_MODEL,
)
from .pdf_utils import process_pdf
from .pptx_utils import process_pptx
from .xlsx_utils import process_xlsx
from .md_utils import process_md

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI()
embedder = SentenceTransformer(EMBED_MODEL)
chroma = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma.get_or_create_collection("doc_chunks")

# ─────────────────────────── helper: push one record ─────────────────────────
def push_record(doc_id: str, part_id: str, caption: str, img_path: pathlib.Path,
                raw_text: str = "") -> None:
    uid = f"{doc_id}_{part_id}"
    record = {
        "doc_id": doc_id,
        "part_id": part_id,
        "caption": caption.strip(),
        "image_path": str(img_path.relative_to(OUT_DIR)),
        "raw_text": raw_text.strip()[:1_000],
    }
    emb = embedder.encode([caption + "\n" + raw_text])[0].tolist()
    collection.add(documents=[caption], embeddings=[emb],
                   ids=[uid], metadatas=[record])
    img_path.with_suffix(".json").write_text(json.dumps(record, indent=2))

# ─────────────────────────── processors dispatcher ───────────────────────────
def process_single(path: pathlib.Path) -> None:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            process_pdf(path, push_record)
        elif ext == ".pptx":
            process_pptx(path, push_record)
        elif ext in {".xlsx", ".xlsm"}:
            process_xlsx(path, push_record)
        elif ext == ".md":
            process_md(path, push_record)
        else:
            print(f"[SKIP] Unsupported file '{path.name}'")
    except Exception:
        traceback.print_exc()

# ───────────────────────── helper: walk a directory ──────────────────────────
def walk_and_process(root_dir: pathlib.Path) -> None:
    files = [p for p in root_dir.rglob("*") if p.is_file()]
    for f in tqdm(files, desc=f"[BATCH] {root_dir.name}", unit="file"):
        process_single(f)

# ─────────────── helper: handle uploaded ZIP / TAR archives ──────────────────
def handle_archive(archive_path: pathlib.Path) -> None:
    tmp_dir = RAW_DIR / f"tmp_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(tmp_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(tmp_dir)
    else:
        raise ValueError("Archive type not recognised")

    walk_and_process(tmp_dir)
    shutil.rmtree(tmp_dir)

# ────────────────────────────── API endpoints ────────────────────────────────
@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    dest = RAW_DIR / file.filename
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())
    process_single(dest)
    return JSONResponse({"status": "processed", "file": file.filename})

@app.post("/batch")
async def ingest_batch(file: UploadFile = File(...)):
    """
    Accepts:
        • A .zip or .tar(.gz) containing an arbitrary tree of docs
        • A browser “directory upload” (files with slashes in filenames)
    """
    dest = RAW_DIR / file.filename
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())

    if dest.is_file() and (zipfile.is_zipfile(dest) or tarfile.is_tarfile(dest)):
        handle_archive(dest)
    else:
        # directory upload: path like 'myDir/file.pdf'
        root = RAW_DIR / dest.stem
        root.mkdir(parents=True, exist_ok=True)
        shutil.move(dest, root / dest.name)
        walk_and_process(root)

    return JSONResponse({"status": "batch processed", "upload": file.filename})

# ────────────────────────────────── main ─────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
