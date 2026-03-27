#!/usr/bin/env python3
"""
SAM2 Server for remote segmentation.

This script sets up a FastAPI server that exposes SAM2 (Segment Anything Model) functionality
as an API endpoint. This allows segmentation to run on a separate machine (potentially with
better GPU resources) while the main application runs elsewhere.

The server supports different versions of SAM2 based on the provided checkpoint and configuration.
"""

import argparse
import base64
import io
import logging
import numpy as np
import os
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from PIL import Image
from typing import List
from pydantic import BaseModel
# Import SAM2 from the correct location
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_log = logging.getLogger(__name__)

# Global SAM2 model
sam_predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    global sam_predictor
    try:
        _log.info("Loading SAM2 predictor...")

        model_path = os.environ.get("SAM_CHECKPOINT", "")
        model_cfg = os.environ.get("SAM_CONFIG", "")

        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"SAM2 checkpoint not found: {model_path}")

        # For SAM2, the config should be in format "configs/sam2.1/sam2.1_hiera_l" (without .yaml)
        # Hydra searches relative to the sam2 package root, so we need "configs/" prefix
        if model_cfg.endswith('.yaml'):
            model_cfg = model_cfg[:-5]  # Remove .yaml extension
        # If it doesn't start with "configs/", add it
        if not model_cfg.startswith('configs/'):
            # Handle both "sam2.1/sam2.1_hiera_l" and just "sam2.1_hiera_l"
            if '/' not in model_cfg:
                # Just filename, assume sam2.1
                model_cfg = f"configs/sam2.1/{model_cfg}"
            else:
                # Has directory like "sam2.1/sam2.1_hiera_l"
                model_cfg = f"configs/{model_cfg}"

        _log.info(f"Using SAM2 model from {model_path}")
        _log.info(f"Using SAM2 config: {model_cfg}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _log.info(f"Using device: {device}")

        # Build SAM2 model with checkpoint and config
        # SAM2 uses build_sam2(config_name, checkpoint, device) API
        # The config_name should be just the name (e.g., "sam2.1_hiera_l"), not a path
        _log.info(f"Building SAM2 model with config={model_cfg}, checkpoint={model_path}, device={device}")
        sam_model = build_sam2(model_cfg, model_path, device=device)
        sam_predictor = SAM2ImagePredictor(sam_model)
        _log.info("SAM2 model loaded successfully")

    except Exception as e:
        _log.error(f"Failed to load SAM2 model: {e}")
        _log.error(f"Error details: {str(e)}")
        import traceback
        _log.error(traceback.format_exc())
        raise

    yield

    # Shutdown (cleanup if needed)
    _log.info("Shutting down SAM2 server")


# Create FastAPI app with lifespan handler
app = FastAPI(
    title="SAM2 Segmentation Server",
    description="Remote segmentation using SAM2",
    lifespan=lifespan
)




class SegmentationRequest(BaseModel):
    """Request model for segmentation API."""
    image_base64: str
    boxes: List[List[float]]  # List of [x0, y0, x1, y1] boxes in pixel coordinates


class SegmentationResponse(BaseModel):
    """Response model for segmentation API."""
    masks: List[List[str]]  # Base64-encoded binary masks
    scores: List[float]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if sam_predictor is None:
        raise HTTPException(status_code=503, detail="SAM2 model not loaded")
    return {"status": "healthy", "model": "SAM2 loaded"}


@app.post("/segment")
async def segment(request: SegmentationRequest) -> SegmentationResponse:
    """Segment objects in an image based on bounding boxes."""
    if sam_predictor is None:
        raise HTTPException(status_code=503, detail="SAM2 model not loaded")

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process with SAM2
        _log.info(f"Processing image of size {image.size} with {len(request.boxes)} boxes")

        # Set image and predict masks using SAM2
        sam_predictor.set_image(image)
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(request.boxes),
            multimask_output=False
        )

        # Encode masks as base64 for response
        encoded_masks = []
        for mask_batch in masks:
            batch_encoded = []
            for mask in mask_batch:
                # Compress mask using run-length encoding
                mask_bytes = io.BytesIO()
                np.save(mask_bytes, mask)
                mask_bytes.seek(0)
                batch_encoded.append(base64.b64encode(mask_bytes.read()).decode())
            encoded_masks.append(batch_encoded)

        # Flatten scores if they have an extra dimension (N, 1) -> (N,)
        scores_flat = scores.flatten().tolist() if scores.ndim > 1 else scores.tolist()

        return SegmentationResponse(masks=encoded_masks, scores=scores_flat)

    except Exception as e:
        _log.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


def main():
    """Run the SAM2 segmentation server."""
    parser = argparse.ArgumentParser(description="SAM2 Segmentation Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM2 checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="SAM2 config name (e.g., 'sam2.1_hiera_l', 'sam2.1_hiera_b', 'sam2.1_hiera_s', 'sam2.1_hiera_t' or full path like 'configs/sam2.1/sam2.1_hiera_l')"
    )
    args = parser.parse_args()

    # Set environment variables for model loading
    os.environ["SAM_CHECKPOINT"] = args.checkpoint
    os.environ["SAM_CONFIG"] = args.config

    # Start server
    _log.info(f"Starting SAM2 server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
