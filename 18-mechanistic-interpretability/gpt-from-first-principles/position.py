'''
Positional embedding can also be thought of as a lookup table, 
but rather than the indices being our token IDs,
the indices are just the numbers 0, 1, 2, ..., seq_len-1 
(i.e. the position indices of the tokens in the sequence).
'''
from jaxtyping import Float, Int

import torch
import torch.nn as nn
from torch import Tensor

from transformer_lens import HookedTransformer

from config import Config
from tests import rand_int_test, load_gpt2_test

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln = False, center_unembed = False, center_writing_weights = False)
device = "cuda" if torch.cuda.is_available() else "cpu"

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model))) # (seq len, model dim)
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        # batch, seq_len = tokens.shape
        # pos_emb = self.W_pos[:seq_len]         # (seq_len, d_model)
        # pos_emb = pos_emb.unsqueeze(0)         # (1, seq_len, d_model)
        # pos_emb = pos_emb.expand(batch, -1, -1)  # (batch, seq_len, d_model)
        # return pos_emb

        # nicer way
        return self.W_pos[torch.arange(0, len(tokens))]

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)

rand_int_test(PosEmbed, [2, 4])
load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
