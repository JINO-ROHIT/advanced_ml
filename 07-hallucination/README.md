### 
What makes an LLM Hallucinate?
Hallucination is where an LLM produces content that is inconsistent with real-world facts or user inputs. 

How does hallucination sneak in?

- Misinformation and biases
 - LLMs mimics the training distribution and if the training set has inaccuracies, they learn the same. This is called imitative falsehood.
 - Biases where LLMs might associate the profession of nursing with females.

- Knowledge Boundary
 - Domain Knowledge Deficiency where LLMs despite being good at text generation lack at being experts in domain data unless explicitly trained on them.
 - Outdated Factual Knowledge where once these models are trained, their internal knowledge is never updated and hence stay outdated.

- Knowledge Shortcut
 - LLM displays a tendency to overly depend on positional close words within the pretraining data and tend to take shortcuts ie when queried about "the capital of Canada", the model erroneously responds with "Toronto". This mistake might arise due to a higher co-occurrence frequency of Canada and Toronto in its training data, leading the model to incorrectly capture the factual knowledge about Canada’s capital.

- Exposure Bias
 - Autoregressive LLMs during inference sample tokens one by one, and they use these generated tokens for the next word prediction. But if an an erroneous token is generated, a whole snowball effect takes place.

All considered, LLM is not a fact learning model and have no concept of facts/knowledge, so essentially all they do is hallucinate, and by hallucinate i mean trying to make a guess from a learned pattern, maybe?


Paper - https://arxiv.org/abs/2311.05232