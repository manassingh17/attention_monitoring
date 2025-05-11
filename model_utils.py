import torch
import joblib
import numpy as np
from PIL import Image
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

clf = joblib.load("attention_classifier.pkl")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_bytes):
    img = Image.open(image_bytes).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(img).squeeze().cpu().numpy().reshape(1, -1)
    proba = clf.predict_proba(emb)[0]
    label = "Attentive" if np.argmax(proba) == 0 else "Inattentive"
    confidence = float(np.max(proba)) * 100
    return {"label": label, "confidence": round(confidence, 2)}
