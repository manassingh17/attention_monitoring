# train.py (updated)
import joblib

# Load features
X, y = joblib.load("embeddings.pkl")

# Just save for inference
joblib.dump((X, y), "attention_memory.pkl")
print("Saved raw embeddings to attention_memory.pkl")
