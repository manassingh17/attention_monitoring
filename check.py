import joblib
import numpy as np

X_ref, y_ref = joblib.load("attention_memory.pkl")
print("Number of samples:", len(y_ref))
print("Attentive samples:", np.sum(y_ref == 0))
print("Inattentive samples:", np.sum(y_ref == 1))
