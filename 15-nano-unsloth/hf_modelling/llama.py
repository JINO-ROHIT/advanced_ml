import torch
import torch.nn as nn

from config import LlamaConfig

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        """reason for float32 upcast here - https://github.com/huggingface/transformers/pull/17437"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

### rotary positional embeddings

def _compute_default_rope_parameters(
        dim: int,
        base: int = 10_000,
        device: str = 'cuda',
):
    # compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype = torch.int64).to(device = device, dtype = torch.float) / dim))
    return inv_freq, 1

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config : LlamaConfig, device = None):
        self.rope_type = 'default' # we are implementing only the default
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        _partial_rotary_factor = 1.0
        _head_dim = config.hidden_size // config.num_attention_heads
        _dim = int(_head_dim * _partial_rotary_factor)
        inv_freq, self.attention_scaling = _compute_default_rope_parameters(_dim, config.rope_theta)
        self.register_buffer("inv_freq", inv_freq, persistent = False) # dont add as a part of state dict
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


### mlp 

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU # check config file

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj