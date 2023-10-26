import pickle
import re
from pathlib import Path
import neptune

classes = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]

__version__ = "0.1.0"
__DIR__     = Path(__file__).resolve(strict=True).parent

with open(f"{__DIR__}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)

def predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()

    pred = model.predict([text])

    # Neptune
    run = neptune.init_run(
        project="stefano-tumino/Language-Prediction",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOTkzZTMwZS1jOTZmLTQzOTYtYTYxYS0xZTQ0OTJmMTcxY2IifQ==",
    )

    run["text"] = text
    run["prediction"] = pred
    
    run.stop()

    return classes[pred[0]]
