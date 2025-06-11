import torch

torch_square = torch.square
torch_mean   = torch.mean

def fast_rms_layernorm_inference(self, X, XX = None, XX2 = None, variance = None):
    """
    1. XX and XX2 are buffers, so instead of reallocating tensors of the ops, reuse these buffers.
    2. torch_mean and torch_square uses in-inplace ops, more efficient.
    Read this for when to use in-place ops https://discuss.pytorch.org/t/when-inplace-operation-are-allowed-and-when-not/169583/3
    """
    old_dtype = X.dtype
    if XX is None:
        XX = X.to(torch.float32)
        variance = XX.square().mean(-1, keepdim = True)
    else:
        XX.copy_(X)
        torch_mean(torch_square(XX, out = XX2), -1, keepdim = True, out = variance)
    variance += self.variance_epsilon
    XX *= variance.rsqrt_()

    if XX is None: X = XX.to(old_dtype)  # noqa: E701
    else: X.copy_(XX)  # noqa: E701

    X *= self.weight
    return X


### rotary positional embeddings

class LlamaRotaryEmbedding(torch.nn.Module):
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(self, dim = None, max_position_embeddings=2048, base=10000, device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Dynamic RoPE we first set it to a max of 4 * 8192 tokens then we iteratively grow this
        self.current_rope_size = min(4 * 8192, self.max_position_embeddings)

        self._set_cos_sin_cache(seq_len=self.current_rope_size, device=device, dtype=torch.get_default_dtype()) # cache embeddings unlike hf

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float() / self.dim)
        )
        t = torch.arange(self.current_rope_size, device="cpu", dtype=torch.int64).float()

        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=dtype, device=device, non_blocking=True), persistent=False) # dont add as a part of state dict
        self.register_buffer("sin_cached", emb.sin().to(dtype=dtype, device=device, non_blocking=True), persistent=False) # dont add as a part of state dict

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype = x.dtype),
            self.sin_cached[:seq_len].to(dtype = x.dtype),
        )


    def get_cached(self, seq_len = None):
        return self.cos_cached, self.sin_cached


    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size: return  # noqa: E701
        # Iteratively grow by increments of 8192
        self.current_rope_size = ((seq_len // 8192) + ((seq_len % 8192) != 0)) * 8192
        self._set_cos_sin_cache(self.current_rope_size, device = "cuda", dtype = x.dtype)