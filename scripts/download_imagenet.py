from datasets import load_dataset, load_from_disk, Dataset
import os
from PIL import Image
import io
import json
import shutil
from tqdm import tqdm

def load_or_download_imagenet(output_path="./datasets/256"):
    if os.path.exists(output_path):
        try:
            print("Found existing dataset, loading from disk...")
            return load_from_disk(output_path)
        except Exception as e:
            print(f"Error loading existing dataset: {e}")
            print("Will download fresh dataset...")
    
    os.makedirs(output_path, exist_ok=True)
    
    ds = load_dataset(
        "benjamin-paine/imagenet-1k-256x256",
        streaming=True
    )
    
    validation_ds = ds["validation"]
    validation_examples = []
    
    with tqdm(desc="Downloading examples") as pbar:
        for example in validation_ds:
            validation_examples.append(example)
            if len(validation_examples) % 100 == 0:
                pbar.update(100)
    
    final_ds = Dataset.from_list(validation_examples)
    final_ds.save_to_disk(output_path)
    return final_ds

def create_imagenet_folders(dataset, output_base_path, class_mapping_path=None):
    os.makedirs(output_base_path, exist_ok=True)

    try:
        if class_mapping_path and os.path.exists(class_mapping_path):
            with open(class_mapping_path, "r") as f:
                imagenet_classes = json.load(f)
        else:
            imagenet_classes = {i: f"class_{i:04d}" for i in range(1000)}
    except Exception:
        imagenet_classes = {i: f"class_{i:04d}" for i in range(1000)}
    
    for class_id in range(1000):
        class_path = os.path.join(output_base_path, imagenet_classes[class_id])
        os.makedirs(class_path, exist_ok=True)
    
    with tqdm(total=len(dataset), desc="Processing images") as pbar:
        for idx, example in enumerate(dataset):
            try:
                image = example["image"]
                label = example["label"]
                
                if not isinstance(image, Image.Image):
                    image = Image.open(io.BytesIO(image["bytes"]))
                
                image_path = os.path.join(
                    output_base_path, 
                    imagenet_classes[label], 
                    f"image_{idx:08d}.jpg"
                )
                image.save(image_path, quality=95)
                pbar.update(1)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")

def cleanup_arrow_dataset(arrow_path):
    try:
        shutil.rmtree(arrow_path)
        print("Cleaned up Arrow dataset")
    except Exception as e:
        print(f"Error cleaning up: {e}")

if __name__ == "__main__":
    ARROW_PATH = "./datasets/256"
    IMAGES_PATH = "./datasets/imagenet/val2"
    CLASS_MAPPING_PATH = "imagenet_classes.json"
    
    dataset = load_or_download_imagenet(ARROW_PATH)
    create_imagenet_folders(dataset, IMAGES_PATH, CLASS_MAPPING_PATH)
    
    if input("Remove Arrow dataset? (y/n): ").lower() == 'y':
        cleanup_arrow_dataset(ARROW_PATH)