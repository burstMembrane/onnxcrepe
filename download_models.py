# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests>=2.32.3",
#     "tqdm>=4.67.1",
# ]
# ///

# https://github.com/yqzhishen/onnxcrepe/releases/download/v1.1.0/full.onnx
# https://github.com/yqzhishen/onnxcrepe/releases/download/v1.1.0/large.onnx
# https://github.com/yqzhishen/onnxcrepe/releases/download/v1.1.0/medium.onnx
# https://github.com/yqzhishen/onnxcrepe/releases/download/v1.1.0/small.onnx
#https://github.com/yqzhishen/onnxcrepe/releases/download/v1.1.0/tiny.onnx

import requests
import tqdm
from pathlib import Path

models_dir = Path("onnxcrepe/assets")
models_dir.mkdir(exist_ok=True)

BASE_URL = "https://github.com/yqzhishen/onnxcrepe/releases/download/v1.1.0"

for model_name in tqdm.tqdm(["full", "large", "medium", "small", "tiny"], desc="Downloading models"):
    url = f"{BASE_URL}/{model_name}.onnx"

    if not Path(models_dir / f"{model_name}.onnx").exists():    
        response = requests.get(url)
        with open(models_dir / f"{model_name}.onnx", "wb") as f:
            f.write(response.content)
