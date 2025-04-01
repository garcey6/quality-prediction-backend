import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
from app.services.data_service import preprocess_data
import logging
import pandas as pd

MODEL_PATH = "models/rnn_model.pth"

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

def train_rnn(data):
    df = pd.DataFrame(data["data"], columns=data["columns"])
    target = data["target"]
    X, y = preprocess_data(df, target), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RNNModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
        loss.backward()
        optimizer.step()
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info("Model saved successfully at %s", MODEL_PATH)
    
    return {"message": "Model trained and saved successfully"}