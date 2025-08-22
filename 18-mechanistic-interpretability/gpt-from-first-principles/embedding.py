from jaxtyping import Float, Int

import torch
import torch.nn as nn
from torch import Tensor

from transformer_lens import HookedTransformer

from config import Config
from tests import rand_int_test, load_gpt2_test

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln = False, center_unembed = False, center_writing_weights = False)
device = "cuda" if torch.cuda.is_available() else "cpu"

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)

rand_int_test(Embed, [2, 4])
load_gpt2_test(Embed, reference_gpt2.embed, tokens)