import torch
from torchvision import models
import torch.nn as nn
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
num_classes = 10
model.fc = torch.nn.Linear(num_features, num_classes)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.load_state_dict(torch.load("../model/model.pt", map_location=device))
model.eval()
model = model.to(device)
