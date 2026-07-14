
import os
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import cv2
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.data = [f for f in glob.glob(folder_path + "/*") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.img_dim = (640, 640)
        self.mean = torch.tensor([0.485, 0.456, 0.406])  # Normalize using ImageNet means
        self.std = torch.tensor([0.229, 0.224, 0.225])   # Normalize using ImageNet stds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = cv2.imread(img_path)

        # Check if the image was loaded correctly
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        # Resize and convert to float32
        img = cv2.resize(img, self.img_dim).astype('float32')

        # Normalize values from [0, 255] to [0, 1]
        img /= 255.0

        # Convert to tensor and permute dimensions
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        # Normalize using mean and std
        img_tensor = (img_tensor - self.mean[:, None, None]) / self.std[:, None, None]

        return img_tensor


# Paths to style and content datasets
content_path = 'C:/Users/anduw/Desktop/big_vision/vision-transformer-pytorch-main/3387817.jpg'
style_path = 'C:/Users/anduw/Desktop/big_vision/vision-transformer-pytorch-main/3388454.jpg'

# Create datasets
content_dataset = CustomDataset(content_path)
style_dataset = CustomDataset(style_path)

# Create DataLoaders
content_loader = DataLoader(content_dataset, batch_size=1, shuffle=True, num_workers=0)
style_loader = DataLoader(style_dataset, batch_size=1, shuffle=True, num_workers=0)



'''
# Example usage to iterate through style and content loaders
for images in style_loader:
    print(f"Style - Batch of images has shape: {images.shape}")
    break

for images in content_loader:
    print(f"Content - Batch of images has shape: {images.shape}")
    break
'''

