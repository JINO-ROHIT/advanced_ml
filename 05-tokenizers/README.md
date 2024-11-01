### Subword Tokenization

Subword tokenization algorithms rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords. For instance "annoyingly" might be considered a rare word and could be decomposed into "annoying" and "ly". Both "annoying" and "ly" as stand-alone subwords would appear more frequently while at the same time the meaning of "annoyingly" is kept by the composite meaning of "annoying" and "ly".

Subword tokenization allows the model to have a reasonable vocabulary size while being able to learn meaningful context-independent representations. In addition, subword tokenization enables the model to process words it has never seen before, by decomposing them into known subwords.

### Why not character level or word level tokenization?

- Character level makes splits at the character level, this means you will never have an out of vocab issue(mostly if the training corpus is large enough)
- But the characters themselves are not useful to the model, the model uses the corresponding emebedding instead.
- Now think about the quality of these embeddings? Consider two words - fly and cry, they differ by a single character but are entirely different words. With the embeddings that you have with character tokenization, the model takes extremely long to converge.

- Word based tokenization, split on words. The notion of words can differ for different languages.
- The quality of embeddings learned can be quite good.
- But we can have infinite number of words! , worst case Dog can be dog, DOG, dOG, DOg etc
