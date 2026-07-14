
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from unet_parts import *
from unet_model import Encoder, Decoder, UNet
from dif_attention import GSA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the encoder and decoder
# Define parameters
n_channels = 3  # For RGB images
n_classes = 3
bilinear=False

# Instantiate the ConvEncoder
encoder = Encoder(n_channels, bilinear)
decoder = Decoder(n_classes, bilinear)

encoder.to(device)
decoder.to(device)

rel_pos_length = 16  # Example value, adjust as needed
dim = 1024
gsa = GSA(dim, rel_pos_length=rel_pos_length, dim_out=1024, heads=8, dim_key=64, norm_queries=True, batch_norm=True).to(device)

#########################################################################################################

#########################################################################################################

import torch
import numpy as np
import cv2
import random

'''
Explanation:
Mask Generation:

We generate the mask by thresholding the attention map. This is based on pixel intensity in the embeddings. 
If a pixel's value in the embeddings is above a threshold, it's considered important and will be included in the mask.
Random Patch Selection:

We extract a random patch from the original image based on the mask. 
The patch size is fixed at [640, 640] (height and width), but it can start at any position where the mask indicates a region of interest.
We ensure that the patch is within the image boundaries, and that its size is [640, 640].
Process:

The process function first generates the mask using the embeddings, 
and then extracts a random patch from the original image based on the generated mask.

'''

import torch
import numpy as np
import random

import torch
import numpy as np

class IntensityDensityPatchSelector:
    def __init__(self, patch_size=(640, 640), threshold=0.5):
        """
        Class to generate a mask based on pixel intensity and density.
        
        Args:
            patch_size (tuple): Size of the patch to be cropped from the original image (height, width).
            threshold (float): Threshold to binarize the mask.
        """
        self.patch_size = patch_size  # Desired output patch size (height, width)
        self.threshold = threshold

    def compute_density(self, mask, kernel_size=5):
        """
        Compute the density of the mask using a convolution with a kernel of ones.
        
        Args:
            mask (torch.Tensor): Binary mask (shape: B x 1 x H x W).
            kernel_size (int): Size of the kernel for density computation.
        
        Returns:
            density (torch.Tensor): Density map (shape: B x 1 x H x W).
        """
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
        density = torch.nn.functional.conv2d(mask, kernel, padding=kernel_size // 2)
        return density

    def generate_mask(self, embeddings, original_image):
        """
        Generate a mask both in embedding space and at the original image level.
        
        Args:
            embeddings (torch.Tensor): Embeddings from the encoder (shape: B x C x H x W).
            original_image (torch.Tensor): Original image (B x 3 x H x W).
        
        Returns:
            embedding_mask (torch.Tensor): Mask in the embedding space (shape: B x 1 x H_emb x W_emb).
            image_level_mask (torch.Tensor): Resized mask at the original image level (shape: B x 1 x H_img x W_img).
        """
        # Generate the embedding-level mask
        embedding_mask = (embeddings.mean(dim=1, keepdim=True) > self.threshold).float()  # Shape: B x 1 x H_emb x W_emb
        
        # Resize the embedding mask to match the original image dimensions
        B, _, H_img, W_img = original_image.shape
        image_level_mask = torch.nn.functional.interpolate(embedding_mask, size=(H_img, W_img), mode="bilinear", align_corners=False)
        
        return embedding_mask, image_level_mask

    def extract_high_intensity_density_patch(self, image, mask):
        """
        Extract the highest intensity and density patch from the original image based on the generated mask.
        
        Args:
            image (torch.Tensor): Original image (B x 3 x H x W).
            mask (torch.Tensor): Mask to guide patch extraction (B x 1 x H x W).
        
        Returns:
            patch (torch.Tensor): Extracted patch (B x 3 x patch_height x patch_width).
        """
        B, C, H, W = image.shape

        # Compute density map
        density_map = self.compute_density(mask)
        
        # Resize the mask to match the original image size if necessary
        density_resized = torch.nn.functional.interpolate(density_map, size=(H, W), mode="bilinear", align_corners=False)
        density_resized = density_resized.squeeze().cpu().numpy()  # Convert to numpy array
        
        # Get the coordinates of the mask regions (where the mask is non-zero)
        mask_coords = np.column_stack(np.where(density_resized > 0))  # y, x coordinates

        # If no valid regions found, return None
        if len(mask_coords) == 0:
            return None

        # Randomly select a starting point from the high-intensity regions
        random_idx = np.random.choice(len(mask_coords))
        start_h, start_w = mask_coords[random_idx]

        # Ensure the patch fits within the image boundaries
        max_h = H - self.patch_size[0]
        max_w = W - self.patch_size[1]
        start_h = min(start_h, max_h)
        start_w = min(start_w, max_w)

        # Extract the patch from the image
        patch = image[:, :, start_h:start_h + self.patch_size[0], start_w:start_w + self.patch_size[1]]
        return patch

    def process(self, embeddings, original_image):
        """
        End-to-end process: generate a mask and apply it to a high-intensity and high-density patch of the original image.
        
        Args:
            embeddings (torch.Tensor): Embeddings from the encoder (shape: B x C x H x W).
            original_image (torch.Tensor): Original image (B x 3 x H x W).
        
        Returns:
            cropped_img (torch.Tensor): Cropped image with the desired dimensions.
            mask (torch.Tensor): Generated mask.
        """
        # Generate the mask based on the embeddings
        embedding_mask, image_level_mask = self.generate_mask(embeddings, original_image)
        
        # Apply the mask to the high-intensity and high-density patch of the original image
        cropped_img = self.extract_high_intensity_density_patch(original_image, image_level_mask)
        
        return cropped_img, image_level_mask





# quit(0)

# Define the load_image function
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((1280, 1280)),  # Resize to the input size of the model
        # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        # transforms.RandomRotation(10),  # Randomly rotate the image by ±10 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet's mean and std
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img.to(device)


# Load an image
image_path_1 = '/home/andualemwelabo.tulu/expt_01/Attentive_Mix/VID_REAL_0329.jpg'
image_path_2 = '/home/andualemwelabo.tulu/expt_01/Attentive_Mix/3388454.jpg'

img_1 = load_image(image_path_1)
img_2 = load_image(image_path_2)

# Forward pass to obtain latent features
x1, x2, x3, x4, x5 = encoder(img_1)  # Pass img_1 through the encoder

embedding_size = (80, 80)
patch_size = (640, 640)
# Example usage
selector = IntensityDensityPatchSelector(patch_size=patch_size)

# Forward pass through the GSA module
emb_at = gsa(x5)

print('embedding', x5.shape)
print('atentive embedding', emb_at.shape)


# Dynamic Thresholdings
import torch
# 
def single_entropy_otsu(embeddings):
    """
    Input:
        embeddings : 4D PyTorch tensor of shape (B, C, H, W),
                     where B is the batch size, C is the number of channels,
                     H and W are the height and width of the embeddings.
    Output:
        threshold : int, a single threshold value for the first batch.
    """
    batch_size, num_channels, height, width = embeddings.shape

    for b in range(batch_size):  # Iterate over each batch
        # Flatten all channels and dimensions into one array
        embedding = embeddings[b].view(-1)  # Shape: (C * H * W,)
        hist = torch.histc(embedding, bins=256, min=0, max=255)  # Compute histogram
        p = hist / hist.sum()

        # Compute entropy Hn
        non_zero_p = p[p > 0]
        Hn = -(non_zero_p * torch.log(non_zero_p)).sum()

        # Initialize variables
        var_max = 0
        optimal_threshold = 0
        total = embedding.numel()
        sum_val = torch.arange(256, dtype=torch.float32, device=embeddings.device).dot(p)
        
        omega_1 = 0
        mu_k = 0
        Ps = 0
        Hs = 0

        for t in range(256):
            omega_1 += p[t]
            omega_2 = 1 - omega_1
            mu_k += t * p[t]
            mu_1 = mu_k / omega_1 if omega_1 > 0 else 0
            mu_2 = (sum_val - mu_k) / omega_2 if omega_2 > 0 else 0

            Ps += p[t]
            if p[t] > 0:
                Hs -= p[t] * torch.log(p[t])
            
            if Ps > 0 and Ps < 1:
                psi = torch.log(Ps * (1 - Ps)) + Hs / Ps + (Hn - Hs) / (1 - Ps)
                current_var = psi * (omega_1 * mu_1**2 + omega_2 * mu_2**2)
                
                # Check if new maximum is found
                if current_var > var_max:
                    var_max = current_var
                    optimal_threshold = t

        return int(optimal_threshold)

# Example usage

# Apply single entropy Otsu's method
threshold = (single_entropy_otsu(emb_at) * 0.0048)
print(f"Single Threshold: {threshold}")

# quit(0)

# Process to get masked image and mask
# Example usage
patch_size = (640, 640)
threshold = threshold # 0.19
selector = IntensityDensityPatchSelector(patch_size=patch_size, threshold=threshold)

# Process to get cropped image and mask
cropped_img, mask = selector.process(emb_at, img_1)

# print("Masked Image Shape:", image_level_mask.shape if image_level_mask is not None else "No valid patch found")
# print("Mask Shape:", mask.shape)

# Saving the Masked Image

import matplotlib.pyplot as plt
import torch
import os

def save_rgb_image(tensor, folder_path, filename, dpi=300, figsize=(8, 8)):
    """
    Save a tensor as an RGB image.

    Args:
        tensor (torch.Tensor): Tensor of shape [1, 3, H, W] (batch size = 1, 3 channels).
        folder_path (str): Path to the folder where the image will be saved.
        filename (str): Name of the file.
        dpi (int): Dots per inch for the saved image.
        figsize (tuple): Size of the figure for saving.
    """
    # Ensure the tensor has the correct dimensions
    assert tensor.ndimension() == 4 and tensor.size(1) == 3, "Tensor must be [1, 3, H, W] for RGB images."

    # Move tensor to CPU and detach
    tensor = tensor.cpu().detach()

    # Unnormalize the image if it was normalized
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = tensor * std + mean

    # Clip values to [0, 1]
    tensor = tensor.clamp(0, 1)

    # Convert tensor to numpy array and transpose to [H, W, C]
    img = tensor[0].permute(1, 2, 0).numpy()

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the full save path
    save_path = os.path.join(folder_path, filename)

    # Save the image
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"RGB image saved to {save_path}")


# Save the cropped image (properly unnormalized and saved as RGB)
if cropped_img is not None:
    folder_path = "/home/andualemwelabo.tulu/expt_01/Attentive_Mix/save_img"
    save_rgb_image(cropped_img, folder_path, "cropped_img08.png", dpi=300, figsize=(8, 8))
else:
    print("No valid patch found")















