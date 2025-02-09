from fastapi import APIRouter
from typing import Dict
from fastapi import File, UploadFile
from PIL import Image
import io

from src.ml.inference import Inference

router = APIRouter()

@router.get("/health")
def get_health() -> Dict[str, str]: 
    return {"health" : "ok"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> str:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    engine = Inference()
    return engine(image)