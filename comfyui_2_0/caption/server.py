import io, os, torch, uvicorn
from fastapi import FastAPI, UploadFile
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = os.getenv("MODEL", "blip2-flan-t5-xl")
proc  = Blip2Processor.from_pretrained(f"Salesforce/{model_id}")
model = Blip2ForConditionalGeneration.from_pretrained(
           f"Salesforce/{model_id}", torch_dtype=torch.float16).to(device)
app = FastAPI()

@app.post("/caption")
async def caption(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(device, torch.float16)
    out    = model.generate(**inputs, max_new_tokens=40)
    return {"caption": proc.decode(out[0], skip_special_tokens=True)}

@app.get("/health")
def _(): return {"ok": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
