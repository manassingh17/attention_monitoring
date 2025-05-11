# attention_app.py
import cv2
import streamlit as st
import torch
import joblib
import numpy as np
from PIL import Image
from torchvision import models, transforms
from collections import deque

# Setup
st.title("🧠 Real-time Attention Detection")
FRAME_WINDOW = st.image([])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

# Load classifier
clf = joblib.load("attention_classifier.pkl")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_embedding(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img).squeeze().cpu().numpy()
    return embedding

# OpenCV video
cap = cv2.VideoCapture(0)
buffer = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Could not access webcam.")
        break

    # Predict
    emb = get_embedding(frame).reshape(1, -1)
    proba = clf.predict_proba(emb)[0]
    buffer.append(proba)
    avg_proba = np.mean(buffer, axis=0)

    label = "Attentive" if np.argmax(avg_proba) == 0 else "Inattentive"
    confidence = avg_proba[np.argmax(avg_proba)] * 100

    # Draw on frame
    color = (0, 255, 0) if label == "Attentive" else (0, 0, 255)
    cv2.putText(frame, f"{label} ({confidence:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
