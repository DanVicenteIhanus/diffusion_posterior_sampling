from torch.utils.data import Dataset
import os
from PIL import Image

class SingleImageDataset(Dataset):
    """Dataset that loads a single image from a directory."""
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.image_path = None
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_path = os.path.join(folder_path, f)
                break
        if self.image_path is None:
            raise FileNotFoundError(f"No valid image found in {folder_path}")
        print(f"Loading single image from: {self.image_path}")
        self.img = Image.open(self.image_path).convert('RGB')
        if self.transform:
            self.img = self.transform(self.img)
    
    def __getitem__(self, _):
        return self.img, 0
        
    def __len__(self):
        return 1