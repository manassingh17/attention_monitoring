import cv2
import torch
import joblib
import numpy as np
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet model
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

clf = joblib.load("attention_classifier.pkl")

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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emb = get_embedding(frame).reshape(1, -1)
    pred = clf.predict(emb)[0]
    label = "Attentive" if pred == 0 else "Inattentive"

    cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0) if label == "Attentive" else (0, 0, 255), 2)
    cv2.imshow("Real-time Attention Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
