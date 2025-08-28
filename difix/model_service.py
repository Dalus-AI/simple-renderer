"""
FastAPI server that will run locally and run inference on difix model.

Server is stateful, one instance of the model is maintained and inference is run on that.
At server start, we warm up the model with a few sample images.

The model loading and inference logic used here was taken from Difix3D/src/inference_difix_compilation.py
"""

import os
import io
from typing import Optional

import numpy as np
from PIL import Image
from fastapi.responses import Response
from fastapi import FastAPI, UploadFile, File, HTTPException

from difix_base_model import Difix
from utils import gpu_mem_report


os.environ["TOKENIZERS_PARALLELISM"] = "false"

WIDTH, HEIGHT = 640, 480
PROMPT = "remove degradation"
MODEL_NAME: Optional[str] = None
# MODEL_PATH = "/workspace/parallax_difix/checkpoints/customer_finetunes/clutterbot/difix_model.pkl"
MODEL_PATH = "/data/diffmodel/model_16001.pkl"
TIMESTEP = 199
MV_UNET = False  # ref_image is always None here

# Warmup images
WARMUP_IMAGES = [
    "/gsplat/data/azure_full_test/images/1.jpg",
    "/gsplat/data/azure_full_test/images/2.jpg",
    "/gsplat/data/azure_full_test/images/3.jpg",
    # "/workspace/parallax_difix/difix_test_data/inference/easy.jpg",
    # "/workspace/parallax_difix/difix_test_data/inference/mid.jpg",
    # "/workspace/parallax_difix/difix_test_data/inference/hard.jpg",
]
WARMUP_REPEATS = 3


app = FastAPI(
    title="Local Difix Inference Service",
    description="Local-only FastAPI wrapper around Difix model sampling.",
    version="0.0.1",
)

# Single model instance
_MODEL: Optional[Difix] = None


@app.on_event("startup")
def _startup_load_and_warmup():
    global _MODEL
    # ... (no changes in your startup; keeping as-is)
    print("Loading model...")
    gpu_mem_report("Before model load.")
    _MODEL = Difix(
        pretrained_name=MODEL_NAME,
        pretrained_path=MODEL_PATH,
        timestep=TIMESTEP,
        mv_unet=MV_UNET,
    )
    _MODEL.set_eval()
    gpu_mem_report("After model load.")
    print("Model loaded.")

    print("########### WARMING UP MODEL ##############")
    for img_path in WARMUP_IMAGES:
        for rep in range(WARMUP_REPEATS):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARMUP] Failed to open {img_path} (attempt {rep+1}): {e}")
                continue

            try:
                gpu_mem_report(f"[WARMUP] Before forward pass {os.path.basename(img_path)} rep={rep+1}")
                _ = _MODEL.sample(
                    image,
                    height=HEIGHT,
                    width=WIDTH,
                    ref_image=None,
                    prompt=PROMPT,
                )
                gpu_mem_report("[WARMUP] After forward pass")
            except Exception as e:
                print(f"[WARMUP] Error during sampling for {img_path} (attempt {rep+1}): {e}")

    print("########### WARM UP COMPLETE ##############")


# Helpers for array path
def _pil_from_float01_rgb(arr: np.ndarray) -> Image.Image:
    """Convert (H,W,3) float array in [0,1] to a PIL RGB image."""
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected (H,W,3), got {arr.shape}")
    arr = np.clip(arr, 0.0, 1.0)
    u8 = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(u8, mode="RGB")


def _float01_from_pil(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to float32 (H,W,3) in [0,1]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return (arr.astype(np.float32) / 255.0)


def _load_npy_or_npz(file_bytes: bytes) -> np.ndarray:
    """Accept .npy or .npz. If .npz, take first array. Expect (H,W,3) or (N,H,W,3) float in [0,1]."""
    bio = io.BytesIO(file_bytes)
    try:
        arr = np.load(bio, allow_pickle=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load .npy/.npz: {e}")
    if isinstance(arr, np.lib.npyio.NpzFile):
        if not arr.files:
            raise HTTPException(status_code=400, detail=".npz contains no arrays.")
        arr = arr[arr.files[0]]
    if arr.ndim not in (3, 4) or arr.shape[-1] != 3:
        raise HTTPException(status_code=400, detail=f"Expected (H,W,3) or (N,H,W,3), got {arr.shape}")
    if arr.dtype not in (np.float32, np.float64):
        raise HTTPException(status_code=400, detail=f"Expected float32/float64 array in [0,1], got dtype {arr.dtype}")
    return arr.astype(np.float32, copy=False)


# Allow either standard image ('file') OR numpy array ('npy')
@app.post("/infer", summary="Run Difix sampling on single image or NumPy array")
async def infer(
    file: UploadFile = File(None, description="JPEG/PNG image"),
    # arrays from Renderer.get_views()
    npy: UploadFile = File(None, description=".npy/.npz (H,W,3) or (N,H,W,3) float in [0,1]"),
):
    """
    Accepts a single image file and returns the processed image.
    If a .npy/.npz is provided (matching Renderer.get_views()), returns a .npy
    with the exact same shape/dtype semantics (float32 in [0,1], batch preserved).
    """
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not initialized yet.")

    # NumPy path to be compatible with Renderer.get_views()
    if npy is not None:
        contents = await npy.read()
        arr = _load_npy_or_npz(contents)  # (H,W,3) or (N,H,W,3), float32 [0,1]

        is_batched = (arr.ndim == 4)
        batch = arr if is_batched else arr[None, ...]  # (N,H,W,3)

        outputs = []
        for i in range(batch.shape[0]):
            h_in, w_in = int(batch[i].shape[0]), int(batch[i].shape[1])

            # Build PIL from float, then RESIZE to model-safe size before sampling
            try:
                pil_in = _pil_from_float01_rgb(batch[i])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid array at index {i}: {e}")

            # Ensure model-friendly spatial dims to avoid UNet shape/broadcast errors
            pil_in = pil_in.resize((WIDTH, HEIGHT), Image.BICUBIC)

            try:
                gpu_mem_report("Before forward pass (API npy)")
                out_pil = _MODEL.sample(
                    pil_in,
                    height=HEIGHT,
                    width=WIDTH,
                    ref_image=None,
                    prompt=PROMPT,
                )
                gpu_mem_report("After forward pass (API npy)")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Inference error at index {i}: {e}")

            # Resize the model output BACK to the original (H,W) so the shape matches get_views()
            if (out_pil.width, out_pil.height) != (w_in, h_in):
                out_pil = out_pil.resize((w_in, h_in), Image.BICUBIC)

            out_arr = _float01_from_pil(out_pil)  # (H,W,3) float32 [0,1]
            outputs.append(out_arr)

        out_batch = np.stack(outputs, axis=0)  # (N,H,W,3)
        if not is_batched:
            out_batch = out_batch[0]  # (H,W,3)

        buf = io.BytesIO()
        try:
            np.save(buf, out_batch)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to encode .npy output: {e}")

        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="output.npy"'},
        )

    # Original image path (kept, with a small safety resize before inference)
    if file is None:
        raise HTTPException(status_code=400, detail="Provide either 'file' (image) or 'npy' (.npy/.npz array).")

    # Read the uploaded image into PIL
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Ensure model-friendly size to avoid shape errors on arbitrary uploads
    if (image.width, image.height) != (WIDTH, HEIGHT):
        image = image.resize((WIDTH, HEIGHT), Image.BICUBIC)

    # Run inference
    try:
        gpu_mem_report("Before forward pass (API)")
        output_image = _MODEL.sample(
            image,
            height=HEIGHT,
            width=WIDTH,
            ref_image=None,
            prompt=PROMPT,
        )
        gpu_mem_report("After forward pass (API)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Return the image, encoded as PNG bytes
    buf = io.BytesIO()
    try:
        output_image.save(buf, format="PNG")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to encode PNG output: {e}")

    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    # Bind to localhost only (local service)
    uvicorn.run("model_service:app", host="127.0.0.1", port=8001, reload=True)