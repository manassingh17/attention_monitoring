import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img).squeeze().cpu().numpy()
    return features

data_dir = "dataset"
classes = os.listdir(data_dir)
X, y = [], []

for label, class_name in enumerate(classes):
    folder_path = os.path.join(data_dir, class_name)
    for img_name in tqdm(os.listdir(folder_path), desc=class_name):
        img_path = os.path.join(folder_path, img_name)
        try:
            features = extract_features(img_path)
            X.append(features)
            y.append(label)
        except:
            print(f"Failed on {img_path}")

X = np.array(X)
y = np.array(y)

joblib.dump((X, y), "embeddings.pkl")
print("Saved embeddings to embeddings.pkl")
