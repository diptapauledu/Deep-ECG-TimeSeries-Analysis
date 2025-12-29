import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.model import LSTMAttentionModel


def evaluate_model(model_path, X_test, y_test):
    model = LSTMAttentionModel(input_dim=1, hidden_dim=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        predictions = model(X_test)

    # Convert to numpy
    y_true = y_test.detach().cpu().numpy().reshape(-1)
    y_pred = predictions.detach().cpu().numpy().reshape(-1)

    # 🔑 FIX: make lengths equal
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return rmse, mae, y_true, y_pred
