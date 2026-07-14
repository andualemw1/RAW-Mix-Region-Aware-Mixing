


import torch
import random


class SourceImagePatcher:
    """
    Class to patch a source image into four regions, randomly remove one region, 
    and replace it with a target patch.
    """

    def __init__(self, src_img, tgt_patch):
        """
        Initialize the patcher with the source image and the target patch.
        
        Args:
            src_img (torch.Tensor): Source image of shape [1, 3, H, W].
            tgt_patch (torch.Tensor): Target patch of shape [1, 3, H/2, W/2].
        """
        self.src_img = src_img.clone()
        self.tgt_patch = tgt_patch.clone()
        self.h, self.w = src_img.shape[2], src_img.shape[3]

        # Validate patch size
        assert self.tgt_patch.shape[2] == self.h // 2 and self.tgt_patch.shape[3] == self.w // 2, \
            "Target patch dimensions must be half the height and width of the source image."

    def patch_and_replace(self):
        """
        Randomly removes one region from the source image and replaces it with the target patch.

        Returns:
            torch.Tensor: Modified source image with the target patch integrated.
        """
        # Divide the source image into four regions
        regions = {
            "top-left": self.src_img[:, :, :self.h // 2, :self.w // 2],
            "top-right": self.src_img[:, :, :self.h // 2, self.w // 2:],
            "bottom-left": self.src_img[:, :, self.h // 2:, :self.w // 2],
            "bottom-right": self.src_img[:, :, self.h // 2:, self.w // 2:]
        }

        # Randomly select a region to replace
        region_to_replace = random.choice(list(regions.keys()))

        # Replace the selected region with the target patch
        modified_src = self.src_img.clone()
        if region_to_replace == "top-left":
            modified_src[:, :, :self.h // 2, :self.w // 2] = self.tgt_patch
        elif region_to_replace == "top-right":
            modified_src[:, :, :self.h // 2, self.w // 2:] = self.tgt_patch
        elif region_to_replace == "bottom-left":
            modified_src[:, :, self.h // 2:, :self.w // 2] = self.tgt_patch
        elif region_to_replace == "bottom-right":
            modified_src[:, :, self.h // 2:, self.w // 2:] = self.tgt_patch

        return modified_src, region_to_replace



