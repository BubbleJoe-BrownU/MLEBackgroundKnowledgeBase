# Large Language Models

**Content**

- [Tokenization](##tokenization)
- [Transformer](##transformer)
  - [BERT](###bert)

## Tokenization
- Word-level
- Character-level
- subword-level
	- bpe
	- wordpiece


## Transformer
explain the process how an input sentence become the output

### Python Implementation

#### Self-Attention Head

```python
class SelfAttentionHead(nn.Module):
  def __init__(self, embed_size, head_size, mask, ctx_length, dropou):
    super().__init__()
    self.M_key = nn.Linear(embed_size, head_size, bias=False)
    self.M_query = nn.Linear(embed_size, head_size, bias=False)
    self.M_value = nn.Linear(embed_size, head_size, bias=False)
    self.atten_dropout = nn.Dropout(dropout)
    self.mask = mask
    if self.mask:
      torch.register_buffer(name="tril", tensor.tril(torch.ones(ctx_length, ctx_length)).reshape(1, ctx_length. ctx_length))
      
  def forward(self, x):
    seq_len = x.shape[1]
    embed_size = x.shape[-1]
    
    key = self.M_key(x)
    query = self.M_query(x)
    value = self.M_value(x)
    # Q @ K.T / sqrt(d)
    weight = query @ key.transpose(-2, -1) / embed_size ** 0.5
    if self.mask:
      weight = weight.masked_fill(self.tril[:, seq_len, seq_len] == 0, value=float("-inf"))
    weight = F.softmax(weight, dim=-1)
    weight = self.atten_dropout(weight)
    return weight @ v
```

#### Cross-Attention Head

```python
class CrossAttentionHead(nn.Module):
  def __init__(self, embed_size, head_size, mask, ctx_length, dropou):
    super().__init__()
    self.M_key = nn.Linear(embed_size, head_size, bias=False)
    self.M_query = nn.Linear(embed_size, head_size, bias=False)
    self.M_value = nn.Linear(embed_size, head_size, bias=False)
    self.atten_dropout = nn.Dropout(dropout)
      
  def forward(self, x, context):
    seq_len = x.shape[1]
    embed_size = x.shape[-1]
    
    key = self.M_key(context)
    query = self.M_query(x)
    value = self.M_value(context)
    # Q @ K.T / sqrt(d)
    weight = query @ key.transpose(-2, -1) / embed_size ** 0.5
    weight = F.softmax(weight, dim=-1)
    weight = self.atten_dropout(weight)
    return weight @ v
```



## BERT

## RoBERTa

## SBERT (Sentence BERT)

Without a shared semantic latent space, to predict whether two sentences are semantically similar, we have to **concatenate two sentences and feed it to BERT**, which is inefficient when the number of sentences are large.

map each sentence to a vector space such that semantically similar sentences are close

How to get **sentence embeddings**

- **average** the BERT output layer, best performance in this works
- use the output of `[CLS]` token
- compute a **max**-over-time of the output 

### Training Framework

**classification objective function**: concatenate the sentence embeddings $u$ and $v$ with the element-wise difference $|u-v|$ and multiply it with the trainable weight $W_t \in \mathbb R^{3n \times k}$
$$
o = \text{softmax}(W_t(u, v, |u-v|))
$$
where $n$ is the dimension of the sentence embeddings and $k$ the number of labels.

**regression objective function**: compute the cosine similarity between the two sentence embeddings $u$ and $v$ is computed. We use mean-squared-error loss as the objective function.

**triplet objective function** Given an anchor sentence $a$, a positive sentence $p$, and a negative sentence $n$, triplet loss tunes the network such that the distance between $a$ and $p$ is smaller than the distance between $a$ and $n$. 
$$
max(||s_a - s_p|| - ||s_a - s_n|| + \epsilon, 0)
$$


### Whole Word Masking

### Chinese BERT

## GPT

### Mistral-7b

What are some differences in model architecture between Llama-2 and Mistral?

Mistral-7b leverages several novel architectural design to increase the inference efficiency.

**Grouped Query Attention (GQA)** is an interpolation between multi-head and multi-query attetion with single key and value heads per subgroup of query heads.

![GQA](/Users/chemstriker/Desktop/Screenshot 2024-04-16 at 1.28.35 PM.png)



**Sliding Window Attention (SWA)**, which allows each token to attend to a maximum of $W$ tokens from the previous layer, to handle longer sequences efficiently. Information can propagate forward by up to k x W tokens after K attention layers. In Mistral-7b, the window size is 4096, and the number of layers is 32, resulting in a theoretical attention span of approximately 131K tokens.

![Sliding Window Attention](https://assets-global.website-files.com/6473d8d02a3cf26273f2787d/65487187e0f7a86ec6b246ab_wzxCpUrCnj2f-seah0ka_z34_bIWHpaG10oNg2it0wa_Qpc2XMGgL2sxNej0AtHC0AItzxq0_Ajr6OqWVcKhTe9cLFg7LeLnkBCRKzUDVZhWYdLtxVoxAAZHe25szCAo2j_tjsm9u3cGLCMvNJPF3Ck.png)

```python
def _make_sliding_window_causal_mask(
  input_ids_shape, 
  dtype, 
  device, 
  past_key_values_length, 
  sliding_window
):
  """
  Make causal mask used for sliding window attention
  """
  bsz, tgt_len = input_ids_size
  tensor = torch.full(
    (tgt_len, tgt_len), 
    fill_value=1
    device=device
  )
  mask = torch.tril(tensor, diagonal=0)
  # make the mask banded to account for sliding window
  mask = torch.triu(mask, diagonal=-sliding_window)
  mask = torch.log(mask).to(dtype)
  
  if past_key_values_length > 0:
    mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    
  return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len+past_key_values_length)
```



**Rolling Buffer Cache**, where a fixed attention span is maintained, and cache size is limited using a rolling buffer cache. 

The cache has a fixed size of $W$, and keys and values for each timestep are stored in position $i\%W$​ of the cache.

Past values in the cache are overwritten when the position $i$ is larger than $W$.

This approach reduces cache memory usage by 8x without compromising model quality.

![Rolling Buffer Cache](https://assets-global.website-files.com/6473d8d02a3cf26273f2787d/65487187c0fbee29bdbbff63_W2ldfwwcqOe3RDYHAJX5UBEWpysReuFou7WM4ad7uZjm-2OI35feiJmReCPt2Ad0e5JejM3VycEcdLW0r_Tv64LZ4gdND0ayJC0s8zIMsJqWjEywM9rnQuYe7i8Fwtx3YsciXznp-q6vPcJiPBohliM.png)

**Pre-Fill and Chunking**. When generating a sequence, tokens are predicted one-by-one, and each token depends on the previous ones.

The prompt is known in advance, allowing pre-filling of the $(k, v)$ cache with the prompt.

For larger prompts, they can be chunked into smaller pieces and the cache pre-filled with each chunk. The window size can be selected as the chunk size. The attention mask works over both the cache and the chunk, ensuring the model has access to the required context while maintaining efficiency.

![Prefilling](https://assets-global.website-files.com/6473d8d02a3cf26273f2787d/65487187fbdf76139cd41399_2jpnM4gZMH3ItQS8G6-8EuG77OtubhtnEtoPLB3ynXPkekzvWDSaZoqP4Mbzm3O-76ixijMD4DmjWkYxXYYDqGTphiETK8ePT0fW3wJEURu8HZh8qSxQ7-VP0_U6Ae_ISaiZVfGdpTsJEHgVlcpYBdM.png)

## Attention Mechanism
QKV matrices

### Multihead attention
why do we adopt multihead attention

```python
# an modified example from Mistral model
# project from input states to q, k, v states
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)
# split q, k, v states into multiple smaller vectors (MultiHead)
query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
# calculate the attention weights
attn_weights = query_states @ key_states.transpose(-2, -1) / math.sqrt(self.head_dim)
# apply attention mask
if atten_mask is not None:
  attn_weights = attn_weights + atten_mask
  
attn_weights = F.softmax(attn_weights, dim=-1)
attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
attn_output = attn_weights @ value_states

# merge multi-head output into a single output
attn_output = attn_output.transpose(1, 2) # [bsz, len, num_heads, head_size]
attn_output = attn_output.reshape(-1, -1, self.num_heads*self.head_dim)

# project the attention output
attn_output = self.o_proj(attn_output)
```



### Multi-Query Attention (MQA)

The only difference between MQA and MHA is that, while the number of query heads remain the same, we only have one key head and one value head

```python
def repeat_kv(hidden_states, n_rep):
  """
  This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep)
  """
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep = 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_key_value_heads*n_rep, slen, head_dim)

# an modified example from Mistral model
# project from input states to q, k, v states
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)
# split q, k, v states into multiple smaller vectors (MultiHead)
query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
# duplicate the key and value state s.t. all query heads interact with the same key and value state
key_states = repeat_kv(key_states, self.num_heads)
query_states = repeat_kv(query_states, self.num_heads)
# calculate the attention weights
attn_weights = query_states @ key_states.transpose(-2, -1) / math.sqrt(self.head_dim)
# apply attention mask
if atten_mask is not None:
  attn_weights = attn_weights + atten_mask
  
attn_weights = F.softmax(attn_weights, dim=-1)
attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
attn_output = attn_weights @ value_states

# merge multi-head output into a single output
attn_output = attn_output.transpose(1, 2) # [bsz, len, num_heads, head_size]
attn_output = attn_output.reshape(-1, -1, self.num_heads*self.head_dim)

# project the attention output
attn_output = self.o_proj(attn_output)
```



### Grouped-Query Attention (GQA)



## Mixture of Exports (MoE) Architecture

Some excerpt from the Switch Transformer paper:

Mixture of Experts (MoE) select different parameters for each in-coming  example. The result is a **sparsely-activated** model - with an **outrageous number of parameters**, but a **constant computational cost**.

![MoE Architecture](/Users/chemstriker/Library/Application Support/typora-user-images/Screenshot 2024-04-28 at 5.51.14 PM.png)

### Sparse Routing

The MoE layer takes an input a token representation $x$ and then routes this to the best determined top-$k$ expoerts, selected from a set $\{E_i(x)\}_{i=1}^N$ of $N$ expoerts. The router variable $W_r$ produces logits $h(x) = W_r \cdot x$ which are normalized via a softmax distribution over the available $N$ experts at that layer.  The gate-value for expert $i$ is given by $p_i(x) = \frac {e^{h(x)_i}} {\sum_{i=1}^N e^{h(x)_j}}$.

The top-$k$ gate values are selected for routing the token $x$. If $\tau$ is the set of selected top-$k$ indices then the output computation of the layer is the linearly weighted combination of each expert's computation on the token by the gate value, $y = \sum_{i\in \tau} p_i(x)E_i(x)$

### Implementation (taking Mixtral as a reference)

```python
class MixtralSparseMoeBlock(nn.Module):
  """
  This implementation is
  strictly equivalent to standard MoE with full capacity (no
  dropped tokens). It's faster since it formulates MoE operations
  in terms of block-sparse operations to accomodate imbalanced
  assignments of tokens to experts, whereas standard MoE either
  (1) drop tokens at the cost of reduced performance or (2) set
  capacity factor to number of experts and thus waste computation
  and memory on padding.
  """
  def __init__(self.config):
    super().__init__()
    
    self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
    self.experts = nn.ModuleList(
      [MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)]
    )
    
  def forward(self, hidden_states):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    
    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weigths, self.top_k, dim=-1)
    # routing_weights ~ [batch_size * sequence_length, top_k]
    # selected_expert ~ [batch_size * sequence_length, top_k]
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    
    # cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)
    
    final_hidden_states = torch.zeros(
      (batch_size * sequence_length, hidden_dim), 
      dtype=hidden_states.dtype, 
      device=hidden_states.device
    )
    
    # one hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be solliciated
    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    # expert_mask ~ [num_experts, top_k, batch_size * sequence_length]
    
    # loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
      expert_layer = self.experts[expert_idx]
      idx, top_x = torch.where(expert_mask[expert_idx])
      
      if top_x.shape[0] = 0:
        continue
      
      # in torch it's faster to index using lists than torch tensors
      top_x_list = top_x.tolist()
      idx_list = idx.tolist()
      
      # index the correct hidden states and compute the expert hidden state for 
      # the current expert. We need to make sure to multiply the output hidden
      # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
      current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim) # [num_states_for_this_expert, hidden_dim]
      current_hidden_state = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None] # [num_states_for_this_expert, hidden_dim, 1]
      
      # however index_add_ only take a tensor as the index, hence the top_x instead of top_x_list
      final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits
```

##  LLM Decoding

### Greedy Decoding

```python
def greedy_decoding(hidden_states):
  dist = torch.nn.functional.softmax(hidden_states, dim=-1)
  return torch.argmax(dist, dim=-1)
```


### Temperature

Used in calculating the softmax values. A higher temperature leads to more randomized output, while a lower temperature leads to more fixed output. Set temperature to zero to have fixed output.

```python
def temperature_based_sampling(hidden_states, temperature):
  dist = torch.nn.functional.softmax(hidden_states/temperature, dim=-1)
  m = torch.distributions.Categorical(probs=dist)
  # or 
  # m = torch.distributions.Categorical(logits=hidden_states/temperature)
  return m.sample()
```

### top-k sampling

`torch.topk(logits, dim=-1, k=top_k)`

Select k most probable outputs, recompute the softmax values and sample from the new probability distribution

```python
def top_k_sampling(hidden_states, top_k):
  top_k_values, indices = torch.topk(hidden_states, dim=-1, k=top_k)
  m = torch.distributions.Categorical(logits=top_k_values)
  return torch.gather(indices, -1, m.sample()[..., None])

# if we want the top_k_sampling compatible with other sampling methods
# we can also make top_k_sampling produce another prob distribution
def top_k_sampling(hidden_states, top_k, sampler):
  probs = hidden_states.new_ones(hidden_states.shape) * float("-inf")
  top_k_values, indices = torch.topk(hidden_states, dim=-1, k=top_k)
  probs.scatter_(-1, indices, top_k_values)
  return sampler(torch.nn.functional.softmax(probs, dim=-1))
```

### top-p (nucleus) sampling

`torch.sort()` and `torch.cumsum()`

keep picking the highest probable tokens until the sum of their probabilities is greater than or equal to a predefined value $p$​

```python
def nucleus_sampling(hidden_status, p):
  probs = torch.nn.functional.softmax(hidden_stats, dim=-1)
  sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
  cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
  nucleus = cum_sum_probs < p
  # prepend ones s.t. we have at least one token with cumulative prob less than p
  nucleus = torch.cat([
    nucleus.new_ones(nucleus.shape[:-1] + (1, )), nucleus[..., :-1]
  ], dim=-1)
  # get the log probability and mask out the non-nucleus
  sorted_log_probs = torch.log(sorted_probs)
  sorted_log_probs[~nucleus] = float("-inf")
  sampled_sorted_index = self.sampler(sorted_log_probs)
  return indices.gather(-1, sampled_sorted_index.unsqueeze(-1))
```

## Chat Templates For Chat Models

In a chat context, rather than continuing a single string of text (as is the case with a standard language model), the model instead continues a conversation that consists of one or more messages, each of which includes a **role**, like "user" or "assistant", as well as message text.

Much like tokenization, different models expect very different input formats for chat. This is the reason we added **chat template** as a feature. **Chat templates are part of the tokenizer**. They specify how to convert conversations, represented as lists of messages, into a single tokenizable string in the format that the model expects.

```python
# Let's take a look at the chat template of the mistral model as an example
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
print(tokenizer.chat_template)
# "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

encoded_input = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "what is your favorite condiment?"}, 
        {"role": "assistant", "content": "Well, I am quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I am cooking up in the kitchen!"}, 
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
)
tokenizer.decode(encoded_input)
# '<s> [INST] what is your favorite condiment? [/INST]Well, I am quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I am cooking up in the kitchen!</s> [INST] Do you have mayonnaise recipes? [/INST]'
```

Chat templates are easy to use. Simply build a list of messages, with `role` and `content` keys, and then pass it to the `apply_chat_template()` method. Once you do that, you'll get output that's ready to go! When using chat templates as input for model generation, it's also a good idea to use `add_generation_prompt=True` to add a **generation prompt**.

**generation prompts**. The `add_generation_prompt=True` argument tells the template to add tokens that **indicate the start of a bot response**. For example, the generation prmopts can be `<|im_start|>assistant`. This ensures that when the model generates text it will write a bot response instead of doing something unexpected, like repeating or continuing the user's message. Chat models are still just language models - they are trained to continue text, and chat is just a special kind of text to them. We need to guide them with appropriate control tokens, so they know what they are supposed to do. Not all models require generation prompts. Some models, like BlenderBot and LLaMA, don’t have any special tokens before bot responses. In these cases, the `add_generation_prompt` argument will have no effect.

### Use of chat templates in training

We recommend that you apply the chat template as a preprocessing step for your dataset. After this, you can simply continue like any other language model training task. When training, you should usually set `add_generation_prompt=False`, because the added tokens to prompt an assistant response will not be helpful during training. Let’s see an example:

```python
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"},
    {"role": "assistant", "content": "The sun."}
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."}
]

dataset = Dataset.from_dict({"chat": [chat1, chat2]})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])
```

From here, just continue training like you would with a standard language modelling task, using the `formatted_chat` column.



## RAG

### Framework of RAG

Key takeaways from *Retrieval-Augmented*

cons of LLM: hallucination, outdated knowledge, non-transparent, untraceable reasoning.

Paradigm:

>  Indexing: raw data is cleaned and extracted to a **uniform plain text**. Then the text is segmented into **chunks**, which is then encoded into **vector representation** using an **embedding model**
>
> Retrieval: the query is first transformed into a vector representation by the same embedding model. Then the top K chunks are retrieved according to the similarity scores.
>
> Generation: The posted query and selected document chunks are synthesized into a coherent prompt and passed to the LLM.

Caveats:

> Retrieval Challenges: the retrieval phase often struggle with precision and recall, leading to the selection of misaligned or irrevelant chunks, and the missing of crucial information.
>
> Generation Difficulities: the model may still produce content not supported by the retrieved context
>
> Augmentation Hurdles: Redundant retrieved context might result in repetitive responses. Hard to maintain stylistic and tonal consistency. For complex task, a one-time retrieval might not suffice to acquire adequate context information.

------

**Naive RAG (Retrieve-Read Framework)** use the aforementioned paradigm to augment the text generation of LLMs

**Advanced RAG (Rewrite-Retrieve-Rerank-Read Framework)**: adopts pre-retrieval and post-retrieval strategies. 

Sliding windo

Modular RAG

----

RAG vs Finetuning

|      | RAG                                                          | Finetuning                                                  |
| ---- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| Cons | higher latency                                               | deep customization of  the model's behavior and style       |
| Pros | real-time knowledge, effective utilization of external knowledge sources, high interpretability | computation resources for training, catastrophic forgetting |

-------------

Embedding models

## Auto-regressive sequence generation

definition
non-autogressive sequence generation



## Multi-GPU training/inference



## Instruction Finetuning

## Preference Optimization

### Reinforcement Learning from Human Feedback (RLHF)

### Direct Preference Optimization (DPO)

## Model Alignment



