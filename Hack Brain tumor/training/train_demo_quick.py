import torch
from model import TumorNet

model = TumorNet()
torch.save(model.state_dict(), "../backend/model.pth")

print("Demo model.pth created (not accurate).")
