### mainly taken from https://github.com/meta-llama/llama/blob/main/llama/model.py

from pydantic import BaseModel
import torch
import torch.nn as nn

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
