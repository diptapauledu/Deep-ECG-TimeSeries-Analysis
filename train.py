import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_ecg
from src.model import LSTMAttentionModel


def prepare_data(csv_path):
    segments = preprocess_ecg(csv_path)

    X = segments[:, :-1]
    y = segments[:, -1]

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(csv_path, epochs=20, lr=0.001):
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(csv_path)

    model = LSTMAttentionModel(input_dim=1, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    print("Training started...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "results/ecg_lstm_attention.pt")

    with open("results/loss_history.txt", "w") as f:
        for l in loss_history:
            f.write(f"{l}\n")

    print("Training finished & loss saved.")

    return model, X_test, y_test
