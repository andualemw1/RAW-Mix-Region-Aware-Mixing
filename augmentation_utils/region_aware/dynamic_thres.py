

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


'''
# Example usage
# Apply single entropy Otsu's method
threshold = (single_entropy_otsu(emb_at) * 0.0048)
print(f"Single Threshold: {threshold}")

'''
