import streamlit as st
import cv2
import torch
import joblib
import numpy as np
from torchvision import transforms, models
from PIL import Image
from scipy.spatial.distance import cdist
from collections import deque

# Load ResNet18
from torchvision.models import resnet18, ResNet18_Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

# Load saved embeddings and labels
X_ref, y_ref = joblib.load("attention_memory.pkl")
X_ref = np.array(X_ref)
y_ref = np.array(y_ref)

# Balance reference dataset
min_count = min(np.sum(y_ref == 0), np.sum(y_ref == 1))
att = np.where(y_ref == 0)[0]
inatt = np.where(y_ref == 1)[0]
np.random.shuffle(att); np.random.shuffle(inatt)
X_ref = np.concatenate([X_ref[att[:min_count]], X_ref[inatt[:min_count]]])
y_ref = np.concatenate([y_ref[att[:min_count]], y_ref[inatt[:min_count]]])

# Transforms
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
        return model(img).squeeze().cpu().numpy()

def predict_similarity(embedding, top_k=5):
    dists = cdist([embedding], X_ref, metric="cosine")[0]
    nearest = np.argsort(dists)[:top_k]
    labels = y_ref[nearest]
    pred = np.bincount(labels).argmax()
    confidence = (labels == pred).sum() / top_k
    return pred, confidence * 100

# Streamlit UI
st.title("🧠 Real-Time Attention Detection")
frame_window = st.image([])
status_text = st.empty()
start = st.button("Start Webcam")

if start:
    buffer = deque(maxlen=10)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        emb = get_embedding(frame)
        label_idx, conf = predict_similarity(emb)
        buffer.append((label_idx, conf))

        # Smooth predictions
        labels, confs = zip(*buffer)
        final_label = max(set(labels), key=labels.count)
        avg_conf = np.mean([c for l, c in buffer if l == final_label])

        label = "Attentive" if final_label == 0 else "Inattentive"
        color = "green" if final_label == 0 else "red"

        # Draw label
        display_frame = frame.copy()
        cv2.putText(display_frame, f"{label} ({avg_conf:.1f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if final_label == 0 else (0, 0, 255), 2)

        frame_window.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        status_text.markdown(f"### Status: <span style='color:{color}'>{label} ({avg_conf:.1f}%)</span>", unsafe_allow_html=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
