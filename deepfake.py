# server.py
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import requests
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
from fastapi.middleware.cors import CORSMiddleware

# -------- Model Definition --------
class Densenet121_Builder(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4, use_pretrained=True):
        super(Densenet121_Builder, self).__init__()
        self.base_model = timm.create_model('densenet121', pretrained=use_pretrained)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1024
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

# -------- Load Model Once --------
def load_model():
    try:
        model = Densenet121_Builder(num_classes=2, use_pretrained=False)
        checkpoint = torch.load('finetuned_densenet121_binary.pth', map_location=torch.device('cpu'))

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

model = load_model()

# -------- FastAPI Setup --------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------- Request Schema --------
class ImageRequest(BaseModel):
    imageUrl: str

# -------- Prediction --------
def predict_image(model: nn.Module, image_url: str) -> dict[str, Any]:
    if not image_url.startswith(('http://', 'https://')):
        raise ValueError("Invalid URL format")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download image with retry
    max_retries = 3
    timeout = 10
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, timeout=timeout)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to download image after {max_retries} attempts: {str(e)}")

    if response.headers.get('content-type', '').split('/')[0] != 'image':
        raise ValueError("URL does not point to a valid image")

    try:
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

    return {
        'prediction': {0: 'Fake', 1: 'Real'}[predicted.item()],
        'confidence': float(probabilities[0][predicted.item()].item())
    }

# -------- FastAPI Endpoint --------
@app.post("/predict")
def predict(req: ImageRequest):
    try:
        result = predict_image(model, req.imageUrl)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------- Health Check --------
@app.get("/")
def root():
    return {"status": "Server running"}
