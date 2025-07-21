from markdown_it import MarkdownIt
from pathlib import Path
from bs4 import BeautifulSoup
from PIL import Image
import subprocess, json, io, os
from .caption_api import auto_caption
from .settings import OUT_DIR, RAW_DIR

md = MarkdownIt("commonmark")

def _render_mermaid(code, out_png):
    cmd = ["mmdc", "-o", out_png, "-i", "-"]
    subprocess.run(cmd, input=code.encode(), check=True)

def process_md(path: Path, push_record):
    doc_id = path.stem
    text = path.read_text("utf-8")
    tokens = md.parse(text)
    out_dir = OUT_DIR / doc_id; out_dir.mkdir(parents=True, exist_ok=True)
    img_i = 0
    for i, tok in enumerate(tokens):
        if tok.type == "image":
            src = tok.attrs["src"]; alt = tok.attrs.get("alt", "")
            img_abs = (path.parent / src).resolve()
            p = out_dir / f"img{img_i}.png"
            Image.open(img_abs).save(p)
            para = (tokens[i+2].content if i+2 < len(tokens) and tokens[i+2].type=="inline" else "")
            caption = alt or para or auto_caption(p)
            push_record(doc_id, f"md_img{img_i}", caption, p, raw_text=para)
            img_i += 1
        if tok.type == "fence" and tok.info.strip() == "mermaid":
            p = out_dir / f"mm{img_i}.png"
            _render_mermaid(tok.content, p)
            caption = "Mermaid diagram: " + tok.content.splitlines()[0][:80]
            push_record(doc_id, f"md_mm{img_i}", caption, p, raw_text=tok.content)
            img_i += 1
