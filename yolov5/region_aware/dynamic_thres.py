
# Dynamic Thresholdings
# https://github.com/mtntruong/entropy-otsu/blob/master/implementation/entropy_otsu.m

import torch

class DynamicThresholdGenerator:
    def __init__(self, alpha=0.5):
        """
        Initialize the DynamicThresholdGenerator.

        Args:
            alpha (float): Scaling factor for the variance to adjust the dynamic range of thresholds.
        """
        self.alpha = alpha

    def calculate_threshold(self, embedding):
        """
        Calculate a dynamic threshold based on the mean and variance of the embedding pixels.

        Args:
            embedding (torch.Tensor): Input embedding of shape [1, C, H, W].

        Returns:
            float: Dynamic threshold value between 0 and 1.
        """
        # Ensure the embedding is in the expected shape
        assert len(embedding.shape) == 4 and embedding.shape[0] == 1, \
            "Embedding should have shape [1, C, H, W]"

        # Flatten the embedding (exclude batch dimension)
        flattened = embedding.view(-1)  # Shape: [C * H * W]

        # Compute mean and variance
        mean = flattened.mean().item()
        variance = flattened.var().item()

        # Compute the raw threshold
        threshold = (mean + self.alpha * (variance ** 0.5)) * 255
        print('debug thresholds',threshold)

        # Calculate dynamic threshold using the given interpolation method
        min_raw_threshold = 80
        max_raw_threshold = 280
        min_result = 0.18
        max_result = 0.76

        if threshold < min_raw_threshold:
            return min_result
        elif threshold > max_raw_threshold:
            return max_result
        else:
            # Linearly interpolate for values between min_raw_threshold and max_raw_threshold
            interpolated_value = ((threshold - min_raw_threshold) / (max_raw_threshold - min_raw_threshold)) \
                                 * (max_result - min_result) + min_result
            
            return interpolated_value


'''
# Example usage
embedding = torch.randn(1, 1024, 80, 80)  # Example tensor with shape [1, 3, 224, 224]
threshold_generator = DynamicThresholdGenerator(alpha=0.15)
threshold = threshold_generator.calculate_threshold(embedding)
print("Calculated Threshold:", threshold)

'''



'''
# Example Usage
if __name__ == "__main__":
    # Example input embeddings (B, C, H, W)
    embeddings = torch.rand((2, 3, 256, 256))  # Batch size 2, 3 channels, 256x256 pixels

    # Initialize the dynamic threshold generator
    threshold_generator = DynamicThresholdGenerator()

    # Calculate dynamic thresholds
    dynamic_thresholds = threshold_generator.calculate_threshold(embeddings)

    print("Dynamic Thresholds:", dynamic_thresholds)
'''




'''
def compute_histogram(tensor, bins, min_val, max_val):
    range_step = (max_val - min_val) / bins
    hist = torch.zeros(bins, device=tensor.device, dtype=torch.float32)
    for value in tensor:
        bin_idx = int((value - min_val) // range_step)
        if 0 <= bin_idx < bins:
            hist[bin_idx] += 1
    return hist

def single_entropy_otsu(embeddings):
    batch_size, num_channels, height, width = embeddings.shape

    for b in range(batch_size):  # Iterate over each batch
        embedding = embeddings[b].view(-1).float()  # Flatten and ensure float32 precision

        # Custom deterministic histogram calculation
        hist = compute_histogram(embedding, bins=256, min_val=0, max_val=255)
        p = hist / hist.sum()

        # Compute entropy Hn
        non_zero_p = p[p > 0]
        Hn = -(non_zero_p * torch.log(non_zero_p)).sum()

        # Remaining logic is the same
        var_max = 0
        optimal_threshold = 0
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

                if current_var > var_max:
                    var_max = current_var
                    optimal_threshold = t

        return int(optimal_threshold)

'''







'''

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
        # Convert to float32 to avoid issues with torch.histc
        embedding = embedding.float()
        
        # Compute histogram
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

'''
