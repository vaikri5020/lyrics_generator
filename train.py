import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import LSTMModel

# 1. Load the data saved from prepare.ipynb
data = torch.load("data/processed.pt", weights_only=False)
itos = data["itos"]
X, y = data["X"], data["y"]

print("Vocabulary size:", len(itos))
print("Training examples:", len(X))

# 2. DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 3. Model, optimizer, loss
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMModel(len(itos)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 4. Train
EPOCHS = 50
for epoch in range(EPOCHS):
    total_loss=0
    for batch_x,batch_y in dataloader:
        batch_x,batch_y=batch_x.to(device),batch_y.to(device)
        optimizer.zero_grad()
        output=model(batch_x)
        loss=loss_fn(output,batch_y)
        loss.backward()
        optimizer.step()
        total_loss=total_loss+loss.item()

    print(f"Epoch:{epoch+1},Loss={total_loss:.4f}")  

torch.save({"model": model.state_dict(), "itos": itos}, "models/lstm.pt")      
    