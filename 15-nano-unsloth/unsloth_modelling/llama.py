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
