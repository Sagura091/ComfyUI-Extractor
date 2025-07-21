import json, shutil, argparse, pathlib, re
SRC = pathlib.Path("/mnt/extract")

def run(dst, caption_key="caption"):
    dst = pathlib.Path(dst); dst.mkdir(parents=True, exist_ok=True)
    for meta in SRC.rglob("*.json"):
        rec = json.loads(meta.read_text())
        cap = rec.get(caption_key, "").strip()
        img = meta.with_suffix(".png")
        if not img.exists(): continue
        shutil.copy(img, dst/img.name)
        (dst/img.name).with_suffix(".txt").write_text(re.sub(r"\s+", " ", cap))
    print("copied to", dst)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dst", required=True)
    p.add_argument("--caption-key", default="caption")
    run(**vars(p.parse_args()))
