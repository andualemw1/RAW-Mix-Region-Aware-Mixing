
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

