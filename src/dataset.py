import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Mapowanie kolorów RGB → klasy
COLOR_MAP = {
    (0, 0, 0): 0,              # tło
    (237, 28, 36): 1,          # łuk
    (181, 230, 29): 2,         # spoina
    (0, 162, 232): 3           # kolumna ciepła (sprawdź inną maskę czy go zawiera)
}


def mask_to_class(mask_img, tolerance=5):
    """Zamień maskę RGB na klasową z tolerancją koloru"""
    mask_array = np.array(mask_img)
    class_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)

    for rgb, cls in COLOR_MAP.items():
        # Oblicz różnicę między każdym pikselem a docelowym kolorem
        diff = np.abs(mask_array - np.array(rgb))
        match = np.all(diff <= tolerance, axis=-1)
        class_mask[match] = cls

    return class_mask

class ThermalSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform
        self.filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_name = self.filenames[idx]
        mask_name = image_name.replace(".jpg", "_mask.png")

        # Ścieżki
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Wczytaj i przeskaluj
        image = Image.open(image_path).convert("L").resize(self.image_size)
        mask = Image.open(mask_path).convert("RGB").resize(self.image_size)

        # Przetwarzanie
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask_class = mask_to_class(mask)
        mask_tensor = torch.from_numpy(mask_class).long()

        return image, mask_tensor, image_name
