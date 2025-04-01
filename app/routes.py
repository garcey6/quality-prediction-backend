from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
from app.services.data_service import preprocess_data
from app.services.train_service import train_rnn
from app.services.predict_service import predict_quality

router = APIRouter()

@router.post("/upload/")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = pd.read_csv(StringIO(contents.decode("utf-8")))
        return {"message": "File uploaded successfully", "columns": data.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/train/")
def train_model(data: dict):
    return train_rnn(data)

@router.post("/predict/")
def predict(data: dict):
    return predict_quality(data)