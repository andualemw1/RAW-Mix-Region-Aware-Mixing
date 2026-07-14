
# UNet Base Autoencoder

# https://jackfan.us.kg/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py

import torch
import os
import torch.nn as nn
from unet_parts import *
from unet_model import Encoder, Decoder, UNet
from dif_attention import GSA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Combined Autoencoder Model
class UNet_GSA(nn.Module):
    def __init__(self, n_channels, n_classes, gsa_dim, gsa_heads=8, bilinear=False):
        super(UNet_GSA, self).__init__()
        self.encoder = Encoder(n_channels, bilinear)
        self.gsa = GSA(dim=gsa_dim, heads=gsa_heads)  # GSA expects a single feature map
        self.decoder = Decoder(n_classes, bilinear)

    def forward(self, x):
        # Encode
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        # Apply attention on the bottleneck feature map (x5)
        x5_attn = self.gsa(x5)
        
        # Decode
        logits = self.decoder(x1, x2, x3, x4, x5_attn)
        return logits


# Forward pass to obtain latent features
x1, x2, x3, x4, x5 = encoder(img_1)  # Pass img_1 through the encoder
recon_img = decoder(x1, x2, x3, x4, x5)  # Pass img_2 through the encoder

print(x5.shape)

# Instantiate the GSA module
rel_pos_length = 16  # Example value, adjust as needed
dim = 1024
gsa = GSA(dim, rel_pos_length=rel_pos_length, dim_out=1024, heads=8, dim_key=64, norm_queries=True, batch_norm=True).to(device)

# Forward pass through the GSA module
emb_at = gsa(x5)
mask = torch.ones(x5.shape)

print(emb_at.shape)

# Print the shape of the latent representation
print(f"Latent shape of image 1: {x5.shape}")
print(f"Recon shape of image 2: {recon_img.shape}")


# Saving the plot image

import matplotlib.pyplot as plt
import torch
import os

def save_image(tensor, folder_path, filename, channel=0, dpi=300, figsize=(8, 8)):
    # Ensure the tensor has the correct dimensions
    assert tensor.ndimension() == 4, "Tensor should have 4 dimensions [batch_size, channels, height, width]"

    # Move tensor to CPU and detach it
    tensor = tensor.cpu().detach().numpy()

    # Select the specific channel (e.g., channel 0)
    tensor = tensor[0, channel, :, :]  # Select the first batch and specific channel

    # Normalize to [0, 1] range for better visualization
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the full path for saving the image
    save_path = os.path.join(folder_path, filename)

    # Save the image with specified DPI and figure size
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(tensor, cmap='viridis')
    ax.axis('off')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Image saved to {save_path}")


print('embedding', x5.shape)
print('sttented embedding', emb_at.shape)
print('recon image', recon_img.shape)

# quit(0)

# Specify the folder path and filenames
folder_path = "/home/andualemwelabo.tulu/expt_01/Attentive_Mix/save_img"
save_image(x5, folder_path, "x5_image.png", channel=0)  # Save the first channel of x5
save_image(emb_at, folder_path, "emb_at_image.png", channel=0)
save_image(recon_img, folder_path, "recon_img_image.png", channel=0)



