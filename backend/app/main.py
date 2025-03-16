# from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model.mobilenetv4 import ThyroidClassifier
import torch
from fastapi import FastAPI, File, UploadFile
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np

app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load PyTorch Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ThyroidClassifier().to(device)
model.load_state_dict(torch.load("model/mobilenetv4.pth", map_location=device))
model.eval()

# Critical Fix 1: Verify Normalization Parameters
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standard
    std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Critical Fix 2: Convert RGB to BGR if needed
    # img = np.array(img)[:, :, ::-1].copy()  # Uncomment if model expects BGR
    # img = Image.fromarray(img)
    
    return preprocess(img).unsqueeze(0).to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    processed = preprocess_image(image)

    with torch.no_grad():
        outputs = model(processed)
        
        # Debug: Print raw outputs
        print(f"Raw model outputs: {outputs.cpu().numpy()}")
        
        probs = torch.nn.functional.softmax(outputs, dim=1)
        print(f"Softmax probabilities: {probs.cpu().numpy()}")

        conf, pred = torch.max(probs, 1)
        print(f"Softmax probabilities: {probs.cpu().numpy()}")

        print("-------------------------\n")
        # Debug: Print all probabilities
        print(f"All probabilities: {probs.cpu().numpy()}")
        
        print(f"Logits: {outputs.cpu().numpy()}")
        print(f"Probabilities: {probs.cpu().numpy()[0]}")
        print(f"Predicted index: {pred.item()}")
        print(f"Confidence: {conf.item()}")
        print("-------------------------\n")

    # CRITICAL FIX 3: Direct class mapping

    # Critical Fix 3: Verify class mapping

    return { 
        "tirads": int(pred.item() + 1),  # Should map 0→1, 1→2, ..., 4→5
        "confidence": round(conf.item() * 100, 1)
        
    }

    # Update your predict route with these changes
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image = await file.read()
#     processed = preprocess_image(image)

#     with torch.no_grad():
#         outputs = model(processed)
#         probs = torch.nn.functional.softmax(outputs, dim=1)
        
#         # Get TOP 5 predictions
#         top5_probs, top5_classes = torch.topk(probs, 5)
        
#         # Convert to percentages
#         probabilities = (top5_probs[0].cpu().numpy() * 100).round(2)
#         class_indices = top5_classes[0].cpu().numpy()

#     # Create class mapping dictionary
#     tirads_mapping = {
#         0: 1,
#         1: 2, 
#         2: 3,
#         3: 4,
#         4: 5
#     }

#     # Find the actual predicted TIRADS class
#     predicted_index = class_indices[0]
#     predicted_tirads = tirads_mapping[predicted_index]
#     confidence = probabilities[0]

#     return {
#         "tirads": predicted_tirads,
#         "confidence": confidence,
#         "all_predictions": {
#         int(tirads_mapping[i]): float(prob) for i, prob in zip(class_indices, probabilities)
#     }
# }