

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





