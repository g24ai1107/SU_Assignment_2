import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from lid_dataset import load_data
from lid_model import LIDModel

X, y = load_data()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

model = LIDModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()

    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Accuracy
    _, preds = torch.max(outputs, 1)
    acc = accuracy_score(y.numpy(), preds.detach().numpy())

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")


    torch.save(model.state_dict(), "models/lid_model.pth")
print("Model saved!")