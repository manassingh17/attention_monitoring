from fastapi import FastAPI, UploadFile, File
from model_utils import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict")
async def predict_attention(file: UploadFile = File(...)):
    result = predict(file.file)
    return result
