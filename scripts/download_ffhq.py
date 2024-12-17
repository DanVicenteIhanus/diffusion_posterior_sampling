"""
Script to download FFHQ resized to 256x256 "validation" set.
"""

from datasets import load_dataset
import os
from PIL import Image
import io
from tqdm import tqdm

def load_ffhq(output_path="./datasets/ffhq"):
    if os.path.exists(output_path):
        print("Found existing dataset")
        return
    
    os.makedirs(output_path, exist_ok=True)
    ds = load_dataset("merkol/ffhq-256")
    process_images(ds["train"], output_path)

def process_images(dataset, output_path, num_images:int = 1000):
    # Original paper seems to use the first 1000 images of FFHQ
    with tqdm(total=num_images, desc="Processing images") as pbar:
        for idx, example in enumerate(dataset):
            if idx >= num_images:
                break
            try:
                image = example["image"]
                if not isinstance(image, Image.Image):
                    image = Image.open(io.BytesIO(image["bytes"]))
                image_path = os.path.join(output_path, f"image_{idx:08d}.jpg")
                image.save(image_path, quality=95)
                pbar.update(1)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")

if __name__ == "__main__":
    load_ffhq()