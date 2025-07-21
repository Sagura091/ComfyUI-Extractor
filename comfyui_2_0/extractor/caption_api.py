import requests, os, json, io, PIL.Image as Image
from .settings import CAPTION_SERVER

def auto_caption(png_path: str) -> str:
    with open(png_path, "rb") as f:
        r = requests.post(f"{CAPTION_SERVER}/caption", files={"file": f})
    try:
        return r.json()["caption"]
    except Exception:
        return ""
