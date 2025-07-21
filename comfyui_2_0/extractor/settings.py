import os, pathlib
RAW_DIR = pathlib.Path("/mnt/raw")
OUT_DIR = pathlib.Path("/mnt/extract")
TIKA_URL = os.getenv("TIKA_URL", "http://tika:9998")
CAPTION_SERVER = os.getenv("CAPTION_SERVER", "http://caption:5000")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "comfyui")