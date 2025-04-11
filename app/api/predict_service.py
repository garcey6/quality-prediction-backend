import torch
import pandas as pd
from app.services.train_service import RNNModel, MODEL_PATH
from app.services.data_service import preprocess_data
import os

def predict_quality(data):
    df = pd.DataFrame(data["data"], columns=data["columns"])
    X = preprocess_data(df)
    model = RNNModel(X.shape[1])
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(X, dtype=torch.float32)).numpy()
        return {"predictions": predictions.tolist()}
    else:
        return {"error": "Model file not found. Please train the model first."}
