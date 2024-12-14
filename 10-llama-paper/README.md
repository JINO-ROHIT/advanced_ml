## A walkthrough of the main concepts of the llama paper and detailed understanding

[Llama](https://arxiv.org/pdf/2302.13971)

#### Tokenization

Uses the BPE algorithm from SentencePiece where they split all numbers into individual digits, and fallback to bytes to decompose unknown UTF-8 characters.

#### Architecture Changes from Transformer

1. RMSNorm instead of LayerNorm. They we normalize the input of each transformer sub-layer, instead of normalizing the output[apparently improves training stability]
2. SwiGLU instead of ReLU
3. RoPE instead of absolute position embedding.
4. Uses gated linear units which is like an element wise multiplcation of linear layers. Dunno why it works tho lol

#### Efficient operations

1. Improved multi-head attention(xformer) by not storing the attention weights and not computing the key/query scores that are masked.
2. To further improve training efficiency, they reduced the amount of activations that are recomputed during the backward pass with checkpointing by saving the activations that are expensive to compute, such as the outputs of linear layers. This is achieved by manually implementing the backward function for the transformer
   layers, instead of relying on the PyTorch autograd.
3. Overlap the computation of activations and the communication between GPUs over the network (due to all_reduce operations) as much as possible.
