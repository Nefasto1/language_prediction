import pickle
import re
from pathlib import Path
import neptune
import json
import os

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
    # with open(f"{__DIR__}/neptune.json", "r") as f:
    # with open(f"/etc/secrets/neptune.json", "r") as f:
    #     neptune_data = json.load(f)

    run = neptune.init_run(
        project   = os.getenv("PROJECT"),
        api_token = os.getenv("API_TOKEN")
    )

    run["text"] = text
    run["prediction"] = classes[pred[0]]
    
    run.stop()

    return classes[pred[0]]
