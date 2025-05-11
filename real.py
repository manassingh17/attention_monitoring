import cv2
import torch
import joblib
import numpy as np
from torchvision import models, transforms
from PIL import Image
from collections import deque
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 (weights API updated)
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

# Load saved reference embeddings and labels
X_ref, y_ref = joblib.load("attention_memory.pkl")
X_ref = np.array(X_ref)
y_ref = np.array(y_ref)

# Balance the reference dataset (equal attentive & inattentive)
min_class_count = min(np.sum(y_ref == 0), np.sum(y_ref == 1))
att_idxs = np.where(y_ref == 0)[0]
inatt_idxs = np.where(y_ref == 1)[0]

np.random.shuffle(att_idxs)
np.random.shuffle(inatt_idxs)

balanced_idxs = np.concatenate([
    att_idxs[:min_class_count],
    inatt_idxs[:min_class_count]
])

X_ref = X_ref[balanced_idxs]
y_ref = y_ref[balanced_idxs]

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Function: get embedding from frame
def get_embedding(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img).squeeze().cpu().numpy()
    return emb

# Function: predict label using cosine similarity
def predict_similarity(embedding, top_k=5):
    dists = cdist([embedding], X_ref, metric="cosine")[0]
    nearest_idxs = np.argsort(dists)[:top_k]
    nearest_labels = y_ref[nearest_idxs]
    pred = np.bincount(nearest_labels).argmax()
    confidence = (nearest_labels == pred).sum() / top_k
    return pred, confidence * 100

# Optional: visualize reference embeddings using PCA
def visualize_embeddings(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    for label in [0, 1]:
        idxs = np.where(y == label)[0]
        plt.scatter(X_pca[idxs, 0], X_pca[idxs, 1],
                    label='Attentive' if label == 0 else 'Inattentive',
                    alpha=0.6)
    plt.title("PCA of Reference Embeddings")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Set up video
cap = cv2.VideoCapture(0)
buffer = deque(maxlen=10)

print("ℹ️ Press 'q' to quit. Press 'v' to visualize embeddings.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    emb = get_embedding(frame)
    label_idx, conf = predict_similarity(emb)
    buffer.append((label_idx, conf))

    # Smooth predictions over buffer
    labels, confs = zip(*buffer)
    final_label = max(set(labels), key=labels.count)
    avg_conf = np.mean([c for l, c in buffer if l == final_label])

    label_text = "Attentive" if final_label == 0 else "Inattentive"
    color = (0, 255, 0) if final_label == 0 else (0, 0, 255)
    display_text = f"{label_text} ({avg_conf:.1f}%)"

    # Overlay label
    cv2.putText(frame, display_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Real-time Attention Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('v'):
        visualize_embeddings(X_ref, y_ref)

cap.release()
cv2.destroyAllWindows()
