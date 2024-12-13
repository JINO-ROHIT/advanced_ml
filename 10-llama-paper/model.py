### mainly taken from https://github.com/meta-llama/llama/blob/main/llama/model.py

from pydantic import BaseModel
import torch
import torch.nn as nn

from typing import Tuple

class ModelArgs(BaseModel):
    dim: int = 4096 # the embedding dimension
    max_seq_len: int = 2048 


#rms norm over layer norm, rms norm hypothesis we dont need to re-scale.
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # learnable parameter
    
    def _norm(self, x):
        '''
        torch.rsqrt(x) ==> 1/square root(x)
        x.pow(2) --> square of x
        '''
        # you can also use x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


## rotary positional embeddings

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):

    # formula is theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)) # not doing [: (dim // 2)] since its basically the same
    t = torch.arange(end, device=freqs.device) 
    freqs = torch.outer(t, freqs).float() 
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):

    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # seperate the vectors as groups of two and then rotate them

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)