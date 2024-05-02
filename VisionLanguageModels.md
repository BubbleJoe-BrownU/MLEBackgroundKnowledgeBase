# Vision Language Models

## Dual-Encoder Models

### CLIP

How clip maintains a large batch size (s.t. the number of negative examples are enormous)

------

**Image Encoder**

```python
class VisionTransformer:
  def __init__(self, ...):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
    
    scale = width ** -0.5
    self.class_embedding = nn.Parameter(scale * torch.randn(width)) # shape = [width, ]
    self.positional_embedding = nn.Parameter(scale * torch.randn((input_resulution//patch_size) ** 2 + 1, width)) # shape = [grid**2 + 1, width]
    self.ln_pre = LayerNorm(width)
    self.transformer = Transformer(width, layers, heads)
    self.ln_post = LayerNorm(width)
    
    self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
    
  def forward(self, x):
    x = self.conv1(x) # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1) # shape = [*, width, grid**2]
    x = x.permute(0, 2, 1) # shape = [*, grid**2, width]
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) # shape = [*, grid**2+1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)
    
    x = x.permute(1, 0, 2)
    x = self.transformer(x)
    x = x.permute(1, 0, 2)
    
    x = self.ln_post(x[:, 0, :]) # the embedding of the CLS token
    
    if self.proj is not None:
      x = x @ self.proj

    return x
```

- How is an input image processed into a sequence of patches?

The image is passed to a convolution layer with `patch_size` as both the `kernel_size` and `stride`, which does the `patchfy` and `flattening` all at once. 

`[*, channel, height, width] -> [*, output_dim, grid, grid]`, where each patch is a 1-D vector with shape `output_dim`.

- How do we get the image embedding from the output of the image encoder?

If the image encoder is a VisionTransformer, the image embedding corresonds to the `[CLS]` token, i.e., the embedding of the first token in a sequence. We get the `[CLS]` embedding and pass it to a projection layer to produce the final image embedding.

-----

**Text Encoder**

```python
x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
```

- How does the text encoder produce text embeddings?

The embedding corresponding to the `[EOS]` token, i.e., the embedding of the last token (but not a padding token). In CLIP, `[EOS]` token has the highest index number.

-----

**Training Framework**

**loss**

```python
# here the logit_scale is similar to temperature, which initialized to np.log(1 / 0.07) = 2.66
image_features = self.encode_image(image)
image_features = F.normalize(image_features, dim=-1)

text_features = self.encode_text(text)
text_features = F.normalize(text_features, dim=-1)

logits_per_image = logit_scale * image_features @ text_features.T
logits_per_text = logit_scale * text_features @ image_features.T
labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
loss = (
  F.cross_entropy(logits_per_image, labels) + 
  F.cross_entropy(logits_per_text, labels)
) / 2
```



### LiT

uses a frozen pre-trained image encoer for CLIP pre-training
### Alpha-CLIP

### MAE (Masked AutoEncoder) used in FLIP

The mask ratio used in MAE is 75%

Important functions like how to break an image into a grid of patches, how to restructure patches to restore the original image, how to apply random masks

```python
def patchify(self, imgs):
# [N, 3, H, W] -> [N, L, patch_size**2 *3]
	p = self.patch_size
  h = w = imgs.shape[2] // p
  x = imgs.reshape((imgs.shape[0], 3, h, p, w, p))
  x = torch.einsum("nchpwq->nhwpqc", x)
  x = x.reshape((imgs.shape[0], h*w, p*p*3))
  return x

def unpatchify(self, x):
  p = self.patch_size
  h = w = int(x.shape[1]**0.5)
  x = x.reshape((x.shape[0], h, w, p, p, 3))
  x = torch.einsum("nhwpqc->nchpwq", x)
  imgs = x.reshape((x.shape[0], 3, h*q, w*q))
  return imgs

def random_masking(self, x, mask_ratio):
  N, L, D = x.shape # batch, length, dimension
  len_keep = int(L*(1 - mask_ration))
  
  noise = torch.rand(N, L, device=x.device)
  
  # sort noise for each sample
  ids_shuffle = torch.argsort(noise, dim=1) # the indices in the ascending order of noises
  ids_restore = torch.argsort(ids_shuffle, dim=1) # torch.gather(ids_shuffle, ids_restore, axis=1)
  
  # keep the first subset
  ids_keep = ids_shuffle[:, :len_keep]
  x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

  # generate the binary mask: 0 is keep, 1 is remove
  mask = torch.ones([N, L], device=x.device)
  mask[:, :len_keep] = 0
  # unshuffle to get the binary mask
  mask = torch.gather(mask, dim=1, index=ids_restore)
  return x_masked, mask, ids_restore

def forward_encoder(self, x, mask_ratio):
  # embed patches
  x = self.patch_embed(x)
  # add pos embed w/o cls token
  x = x + self.pos_embed[:, 1:, :]
  # masking: length -> length * mask_ratio
  x, mask, ids_restore = self.random_masking(x, mask_ratio)
  # append cls token
  cls_token = self.cls_token + self.pos_embed[:, :1, :]
  cls_tokens = cls_token.expand(x.shape[0], -1, -1)
  x = torch.cat((cls_tokens, x), dim=1)
  # apply Transformer blocks
  for blk in self.blocks:
      x = blk(x)
  x = self.norm(x)

  return x, mask, ids_restore

def forward_decoder(self, x, ids_restore):
  # embed tokens
  x = self.decoder_embed(x)

  # append mask tokens to sequence
  mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
  x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
  x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
  x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

  # add pos embed
  x = x + self.decoder_pos_embed

  # apply Transformer blocks
  for blk in self.decoder_blocks:
      x = blk(x)
  x = self.decoder_norm(x)

  # predictor projection
  x = self.decoder_pred(x)

  # remove cls token
  x = x[:, 1:, :]

  return x

def forward_loss(self, imgs, pred, mask):
  """
  imgs: [N, 3, H, W]
  pred: [N, L, p*p*3]
  mask: [N, L], 0 is keep, 1 is remove, 
  """
  target = self.patchify(imgs)
  if self.norm_pix_loss:
      mean = target.mean(dim=-1, keepdim=True)
      var = target.var(dim=-1, keepdim=True)
      target = (target - mean) / (var + 1.e-6)**.5

  loss = (pred - target) ** 2
  loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

  loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
  return loss

def forward(self, imgs, mask_ratio=0.75):
  latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
  pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
  loss = self.forward_loss(imgs, pred, mask)
  return loss, pred, mask
```




## Encoder-Decoder Models

### BEiT

### Flamingo

inserts new cross-attention layers into the LLM to inject visual features, and pre-train the new layer on billions of image-text pairs.

### BLIP: Bootstrapping Language-Image Pre-training



### BLIP-2: BLIP with Frozen Image Encoders and LLMs

bootstrap vision-language pre-training from off-the-shelf **frozen pre-trained image encoders** and **frozen LLMs (decoder-only OPT or encoder-decoder FlanT5)**. Frozen: to reduce computation cost and mitigate the catastrophic forgetting.

------

**Q-Former** (Querying Transformer): an information bottleneck (it **extracts a fixed number of output features** from the image encoder despite of the image resolution) between the frozen image encoder and the frozen LLM to bridge the modality gap. The Q-former is a lightweight transformer which employs a set of learnable query vectors to extract visual features from the frozen image encoder. It feeds the most useful visual feature for the LLM to output the desired text.

Q-Former consists of two transformer submodules that **share the same self-attention layers**: (1) an image transformer that interacts with the frozen image encoder for visual feature extraction. (2) a text transformer that can function as **both a text encoder and a text decoder**.

The query itself is a fixed amount of learnable embeddings that can interact with each other and/or text features through self-attention layers, and interact with frozen image features through **cross-attention layers** inserted **every other transformer block**.

The Q-Former is initialized with the pre-trained weights of BERT-base, whereas the cross-attention layers are randomly initialized.

-----

**Datasets**: The same pre-training dataset as BLIP with 129M images in total, including COCO, Visual Genome, CC3M, CC12M, SBU, and 115M images from the LAION400M dataset. Used the CapFilt method to create synthetic captions for the web images - generating 10 captions with BLIP-large and rank the synthetic captions along with the original web caption based on the image-text similarity produced by a CLIP ViT-L/14 model, and keep top two captions per image.

**Two-stage bootstrapping**: the first stage bootstraps vision-language representation learning from a frozen image encoder. the second stage bootstraps vision-to-language generative learning from a frozen language model.

**Stage One**: Vision-Language Representation Learning

Enforces the Q-Former to learn visual representation most relevant to the text.

Three Pre-training objectives (With Three **different mask patterns**): 

- **Image-Text Matching** (full mask) aims to learn fine-grained alignment between image and text representation. It is a binary classification task where the model is asked to predict whether an image-text pair is positive (matched) or negative (unmatched). Feed each output query embedding into a two-class linear classifier to obtain a logit, and **average the logits** across all queries as the output matching score. **Hard negative mining**
- **Image-Text Contrastive Learning** (Uni-modal self-attention mask) aims to align image representation and text representation such that their mutual information is maximized. Contrasting the image-text similarity of a positive pair against those of negative pairs. First compute the pairwise similarity between each query output and the text embedding, and then select **the highest one as the image-text similarity**. Use **in-batch negatives** instead of the momentum queue in BLIP because we can fit more samples per GPU due to the use of a frozen image encoder.
- **Image-Grounded Text Generation** (Multi-Modal Causal Self-Attention (prefix causal mask)) The ITG loss trains the Q-Former to generate texts conditioned on the input image. The quries are forced to extract visual features that capture all the information about the text. Similar to UniLM.

**Stage Two**: Vision-to-Language Generative Training

Connects the output of the Q-Former to a frozen LLM, and trains the Q-Former s.t. its output visual representation can be interpreted by the LLM.

In the generative pre-training stage, we connect Q-Former (with the frozen image encoder attached) to a frozen LLM to harvest the LLM's generative language capability. Use a fully-connected (FC) layer to linearly project the output query embeddings Z into the same dimension as the text embedding of the LLM. The projected query embeddings function as **soft visual prompts** that condition the LLM on visual representation extracted by the Q-Former.

-------

### LLAVA

**Architecture**

![Llava Architecture](https://images.prismic.io/encord/b81c019b-5d0a-44eb-8a6c-f6ccb4f3e24a_image4.png?auto=compress,format)

Visual Encoder: CLIP ViT-L/14. Image features are **the feature before the last layer**. CLIP's last layer features may focus more on global and abstract image properties compared to the layer before it, which can focus more on localized properties that are useful for understanding specific image details. How to validate this hypothesis: check the attention scores from each visual token embedding, if the visual token is more 

**A projection layer** to connect image features into the word embedding space. Similar to the gated cross attention in Flamingo and Q-former in BLIP-2.

Text Encoder: Vicuna, an LLM based on Llama2

---------

**Visual Instruction Data Generation**
Naive expansion from image-caption pairs: `Human: X_q X_v <STOP> Assistant: X_q <STOP>`, which lacks diversity and in-depth reasoning.

**Context Type** (passed to a text-only GPT to generate instruction following data):

> 1. Captions
> 2. Bounding Boxes: label and box coordinates

**Response Type**:

> 1. Conversation
> 2. Detailed Description
> 3. Complex Reasoning

------

**two-stage training**: 

Stage 1: Pre-training for **Feature Alignment**

trainable module: the **projection layer**

TL;DR: align the image features with the pre-trained LLM word embedding, i.e., train a visual tokenizer for the frozen LLM.

pre-training stage aimed at only updating the projection matrix, the bridge between CLIP and Vicuna, using a subset of the CC3M dataset. This allows input images and input text to be mapped to the same embedding space, allowing the LLM to understand the context from both image and the input prompt. The projection was designed to be simple and lightweight, allowing for faster iterations during experimentation.

Stage 2: Fine-tuning End-to-end

trainable module: the **projection layer and the LLM**

-----

**Llava 1.5**

1. An **MLP vision-language connector**. A two-layer MLP significantly enhances Llava-1.5's multimodal capabilities.
2. Academic task-oriented data. Integrating VQA datasets that are designed for academic tasks.

-----

Llava 1.6 (Llava-NeXT)

Llava-1.6 considers more LLMs, including Mistral-7B and Nous-Hermes-2-Yi-34B
