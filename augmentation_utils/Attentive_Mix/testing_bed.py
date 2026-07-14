# Test module

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from unet_model import Encoder, Decoder, UNet
from simple_vit import ViT
from masking import Masking, Self_Attn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the image and transform it
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

# Create an instance of the ViT model
vit_model = ViT(
    image_size=640,
    patch_size=320,
    # num_classes=1000,  # Add this line to define the number of classes
    dim=1024,
    depth=12,
    heads=12,
    mlp_dim=3072,
    # pool='cls',
    channels=3,
    dim_head=64,
    dropout=0.1,
    emb_dropout=0.1
)

# Load an image
image_path_1 = 'C:/Users/anduw/Desktop/my_project/Attentive_Mix/TGT_5099.jpg'
image_path_2 = 'C:/Users/anduw/Desktop/my_project/Attentive_Mix/CHS_SSEC1.jpg'

img_1 = load_image(image_path_1)  # Call the load_image function here
img_2 = load_image(image_path_2)  # Call the load_image function here

# Pass the image through the model
vit_model.eval()
with torch.no_grad():
    latent_vector = vit_model(img_1)

print(f"Type of latent_vector: {type(latent_vector)}")
print(f"Shape of latent_vector: {latent_vector.shape}")

# Define parameters
n_channels = 3  # For RGB images
n_classes = 3
bilinear=False

# Instantiate the ConvEncoder
encoder = Encoder(n_channels, bilinear)
decoder = Decoder(n_classes, bilinear)

# Forward pass through the encoder
x1, x2, x3, x4, x5 = encoder(img_1)
y1, y2, y3, y4, y5 = encoder(img_2)

# Print the shape of the latent codes
print(f"Input images shape: {img_1.shape}")
print(f"Latent codes shape: {x5.shape}")

latent_vector = latent_vector.view(1, 1024, 1, 1)

print('brodcasted latent vector', latent_vector.shape)

result = latent_vector + x5

print('result', result.shape)

recon_img = decoder((x1+y1), (x2+y2), (x3+y3), (x4+y4), (x5+y5))

print('recon image', recon_img.shape)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Instantiate the model
# ae_model = AdversarialAE(n_channels=3, latent_dim=256)

# Convert tensor to numpy array and plot the image
def plot_image(tensor):
    # Move the tensor to CPU and detach
    tensor = tensor.cpu().detach().numpy()
    # Reshape the tensor and normalize to [0, 1] range
    tensor = tensor.squeeze(0).transpose(1, 2, 0)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    # Plot the image
    plt.imshow(tensor)
    plt.axis('off')
    plt.show()

# Plot the reconstructed image
# plot_image(recon_img)

# Example usage Masking
in_dim = 1024  # Ensure this is a single integer
activation = 'relu'    # Silu

# Initialize the Self_Attn layer
attention_layer = Self_Attn(in_dim, activation)

# Sample input tensor
z, att_map = attention_layer(x5)

print('embeddings', z.shape)
print('attention map', att_map.shape)

# Initialize the Masking layer with the Self_Attn layer
masking_layer = Masking(dim=1024, attention_module=attention_layer)

# Apply the Masking layer to the encoder output
masked_embeddings, mask, attention_map = masking_layer(x5, mask_ratio=0.5)

print("Masked Embeddings Shape:", masked_embeddings.shape)
print("Mask Shape:", mask.shape)
print("Attention Map Shape:", attention_map.shape)

recon_img_2 = decoder(x1, x2, x3, x4, masked_embeddings)

# Plot the reconstructed image
plot_image(recon_img_2)


