'''
Layer Norm

- Simple normalization function applied at the start of each layer (i.e. before each MLP, attention layer, and before the unembedding).
- Converts each input vector (independently in parallel for each batch x position residual stream vector)
  to have mean zero and variance 1. Then applies an elementwise scaling and translation.

- Cool maths tangent: The scale & translate is just a linear map. LayerNorm is only applied immediately before another linear map.

- LayerNorm is annoying for interpretability - the scale part is not linear,
  so you can't think about different bits of the input independently.
  But it's almost linear - if you're changing a small part of the input it's linear,
  but if you're changing enough to alter the norm substantially it's not linear.

y = (x - E[x]) / sqrt(Var[x] + ε) * γ + β
'''

# Reference - https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

from jaxtyping import Float

import torch
import torch.nn as nn
from torch import Tensor

from transformer_lens import HookedTransformer

from config import Config
from tests import rand_float_test, load_gpt2_test

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln = False, center_unembed = False, center_writing_weights = False)
device = "cuda" if torch.cuda.is_available() else "cpu"

class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))
    
    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        numerator = residual - torch.mean(residual, dim = -1, keepdim = True)
        scale_factor = torch.var(residual + self.cfg.layer_norm_eps, unbiased = False, dim = -1, keepdim = True).sqrt()

        y = ( numerator / scale_factor) * self.w + self.b
        return y

rand_float_test(LayerNorm, [2, 4, 768])

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
logits, cache = reference_gpt2.run_with_cache(tokens)

load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])