from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from .model.model import __version__
from .model.model import predict_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

class Item(BaseModel):
    text: str

@app.get('/')
def home():
    return {"health_check": "OK", "Version": __version__}

@app.post('/predict')
def prediction(paiload: Item):
    return {"text_language": predict_pipeline(paiload.text)}