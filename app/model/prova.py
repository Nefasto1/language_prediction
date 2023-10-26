import json
from pathlib import Path
path = Path(__file__).resolve(strict=True).parent

print(path)
with open(str(path) + "/neptune.json", "r") as f:
    data = json.load(f)

print(data["project"])
print(data["api_token"])