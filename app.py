"""
Model server for index.html.

Put this file next to covid_ct_baseline.py and your saved weights, then run:
    pip install fastapi uvicorn python-multipart matplotlib
    uvicorn app:app --port 8000
Then open index.html in a browser.
"""
import base64
import io
from pathlib import Path

import matplotlib
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from covid_ct import DEVICE, IMG_SIZE, build_model, eval_tf, grad_cam

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS = BASE_DIR / "heree.pt"
THRESHOLD = 0.5                    # replace with the value picked on validation data

model = build_model(pretrained=False)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse(BASE_DIR / "frontenf.html")


def heatmap_png(model, x, base):
    """Grad-CAM overlay as base64 PNG. Returns None if the hook fails."""
    try:
        cam = grad_cam(model, x)
        cam = np.asarray(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        ) / 255.0
        color = matplotlib.colormaps["jet"](cam)[..., :3] * 255
        blend = 0.6 * np.asarray(base.resize((IMG_SIZE, IMG_SIZE))) + 0.4 * color
        buf = io.BytesIO()
        Image.fromarray(blend.astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print("Grad-CAM failed:", e)
        return None


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = eval_tf(img)
    with torch.no_grad():
        prob = torch.sigmoid(model(x.unsqueeze(0).to(DEVICE))).item()
    return {"prob": prob, "threshold": THRESHOLD, "heatmap": heatmap_png(model, x, img)}