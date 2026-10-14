import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the same architecture used during training.
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 1),
)

# Load the saved weights.
state = torch.load(
    "best_model_fold_1.pt",
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state)
model.to(device)
model.eval()

# Match the notebook's evaluation preprocessing.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

image = Image.open("my_ct_image.png").convert("RGB")
image_tensor = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    logit = model(image_tensor).squeeze()
    covid_probability = torch.sigmoid(logit).item()

prediction = "COVID" if covid_probability >= 0.5 else "non-COVID"
print(f"{prediction} (COVID probability: {covid_probability:.3f})")