import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from model import TumorResNet

# --------- Data transforms ----------
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ---------- Dataset ----------
train_set = datasets.ImageFolder("dataset/train", transform=train_tf)
test_set  = datasets.ImageFolder("dataset/test",  transform=test_tf)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=16)

# ---------- Model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = TumorResNet(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------- Train Loop ----------
epochs = 10
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.4f}")

# ---------- Save model ----------
# In your training/saving script
state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    # ... other metadata
}
torch.save(state, 'model_new_checkpoint.pth')
print("✔ model.pth saved successfully!")
