from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .model.model import __version__
from .model.model import predict_pipeline

app = FastAPI()

app.addMiddleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

@app.get('/')
def home():
    return {"health_check": "OK", "Version": __version__}

@app.post('/predict')
def prediction(paiload: str):
    return {"text_language": predict_pipeline(paiload)}