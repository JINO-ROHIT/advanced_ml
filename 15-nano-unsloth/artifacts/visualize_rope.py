import torch

import matplotlib.pyplot as plt

from hf_modelling.llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

device = "cuda" if torch.cuda.is_available() else "cpu"
rotary_emb = LlamaRotaryEmbedding(device=device)

batch_size = 2
num_heads = 32
seq_len = 10
head_dim = 128

# query and key tensors
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device = device)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device = device)

# position IDs (for a sequence of length 10)
position_ids = torch.arange(seq_len, dtype = torch.long, device = device).unsqueeze(0).repeat(batch_size, 1)
# shape: [batch_size, seq_len] = [2, 10]

# cosine and sine components
cos, sin = rotary_emb(q, position_ids)

# apply rotary embeddings to query and key
q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1)

print(q.shape, k.shape, position_ids.shape)

def plot_rotary_embeddings(original, rotated, title_prefix=""):
    """Plot original vs rotated vectors in 3D space."""
    #  first 3 dimensions for visualization
    original = original[0, 0, :, :3].cpu().detach().numpy()  # [seq_len, 3]
    rotated = rotated[0, 0, :, :3].cpu().detach().numpy()   # [seq_len, 3]
    
    fig = plt.figure(figsize=(12, 6))
    
    # original
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(original.shape[0]):
        # use quiver plot for arrow like  
        ax1.quiver(0, 0, 0, # start from origin
                  original[i, 0], # x comp
                  original[i, 1], # y comp
                  original[i, 2], # z comp
                  color = plt.cm.viridis(i / original.shape[0]), 
                  arrow_length_ratio = 0.1, 
                  label = f'Pos {i}')
    ax1.set_title(f"{title_prefix}Original Vectors")
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.legend()
    
    # rotated
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(rotated.shape[0]):
        ax2.quiver(0, 0, 0, 
                  rotated[i, 0], 
                  rotated[i, 1], 
                  rotated[i, 2], 
                  color = plt.cm.viridis(i / rotated.shape[0]), 
                  arrow_length_ratio = 0.1, label = f'Pos {i}')
    ax2.set_title(f"{title_prefix}After Rotary Embedding")
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

print("Query Vectors:")
plot_rotary_embeddings(q, q_embed, "Query ")

# Plot key vectors before and after
print("\nKey Vectors:")
plot_rotary_embeddings(k, k_embed, "Key ")