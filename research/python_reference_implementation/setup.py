# import os
import requests
from pathlib import Path

def download_file(url, target_path):
    """Download a file from URL to target path"""
    print(f"Downloading {url} to {target_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {target_path.name}")

def setup_florence2():
    # Setup paths
    base_dir = Path(__file__).parent / "data"
    model_dir = base_dir / "models"
    tokenizer_dir = base_dir / "tokenizer"
    test_dir = base_dir / "test_data"

    # Model files to download
    model_files = [
        "decoder_model.onnx",
        "embed_tokens.onnx", 
        "encoder_model.onnx",
        "vision_encoder.onnx"
    ]

    # Tokenizer files to download
    tokenizer_files = [
        "vocab.json",
        "added_tokens.json",
        "merges.txt"
    ]

    # Download ONNX models
    model_variant = "base-ft"  # Can be base, base-ft, large, or large-ft
    for model_file in model_files:
        url = f"https://huggingface.co/onnx-community/Florence-2-{model_variant}/resolve/main/onnx/{model_file}?download=true"
        download_file(url, model_dir / model_file)

    # Download tokenizer files
    for tokenizer_file in tokenizer_files:
        url = f"https://huggingface.co/onnx-community/Florence-2-{model_variant}/resolve/main/{tokenizer_file}?download=true"
        download_file(url, tokenizer_dir / tokenizer_file)

    # Download test image
    test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    download_file(test_image_url, test_dir / "car.jpg")

if __name__ == "__main__":
    setup_florence2()