import cv2
import torch
import joblib
import numpy as np
from torchvision import models, transforms
from PIL import Image
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet feature extractor
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

# Load trained classifier
clf = joblib.load("attention_classifier.pkl")

# Preprocessing for each frame
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

# --- Smoothing setup ---
buffer = deque(maxlen=10)  # keeps last 10 probability predictions

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get embedding and predict probability
    emb = get_embedding(frame).reshape(1, -1)
    proba = clf.predict_proba(emb)[0]  # [attentive_prob, inattentive_prob]
    buffer.append(proba)

    # Average probabilities over the buffer
    avg_proba = np.mean(buffer, axis=0)
    label_idx = np.argmax(avg_proba)
    label = "Attentive" if label_idx == 0 else "Inattentive"
    confidence = avg_proba[label_idx] * 100

    # Choose text color
    color = (0, 255, 0) if label == "Attentive" else (0, 0, 255)
    text = f"{label} ({confidence:.1f}%)"

    # Draw on frame
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2)
    cv2.imshow("Real-time Attention Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
