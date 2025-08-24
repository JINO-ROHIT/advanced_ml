from jaxtyping import Float

import torch
import torch.nn as nn
from torch import Tensor

from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new
import einops

from config import Config
from tests import rand_float_test, load_gpt2_test

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln = False, center_unembed = False, center_writing_weights = False)
device = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        pre = einops.einsum(
            normalized_resid_mid, self.W_in,
            "batch position d_model, d_model d_mlp -> batch position d_mlp",
        ) + self.b_in
        post = gelu_new(pre)
        mlp_out = einops.einsum(
            post, self.W_out,
            "batch position d_mlp, d_mlp d_model -> batch position d_model",
        ) + self.b_out
        return mlp_out

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
logits, cache = reference_gpt2.run_with_cache(tokens)

rand_float_test(MLP, [2, 4, 768])
load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])