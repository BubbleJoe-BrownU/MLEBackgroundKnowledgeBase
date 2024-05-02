# Deep Learning



## Activation Functions

### ReLU (Rectified Linear Unit)

### LeakyReLU

Leaky ReLU is a type activation function based on a ReLU, but it has a small slope for negative values instead of a flat slope. The slope coefficient $\alpha$ is a hyperparameter determined before training.
$$
\text{LeakyReLU}(x) =
\begin{cases}
x & \text{if x >= 0} \\
\alpha x & \text{otherwise}
\end{cases}
$$

### Sigmoid (Logistic)

$$
\sigma(x) = \frac {1} {1 + e^{-x}}
$$



### Tanh (Hyperbolic Tangent)

$$
\tanh(x) = \frac {e^x - e^{-x}} {e^x + e^{-x}}
$$

### GLU (Gated Linear Unit)

$$
\text{GLU}(a, b) = a \otimes \sigma(b)
$$

GLU is often used in natural language processing architectures, for example, the Gated CNN, because here $b$ is the gate that control what information from $a$ is passed up to the next layer. Intuitively, for a language modeling task, the gating mechanism allows selection of words of features that are important for predicting the next word. The GLU also has non-linear capabilities, but has a linear path for the gradient so dimishes the vanishing gradient problem.

### SiLU (Sigmoid Linear Unit)

$$
\text{SiLU}(x) = x\sigma(x)
$$

### GELU (Gaussian Error Linear Units)

used in BERT, GPT-3, etc.

`torch.nn.GELU(approximate="none")`

`torch.nn.functional.gelu(input, approximate='none')`

`approximate = none | tanh`

The GELU activation function is $x\Phi(x)$, where $\Phi(x)$ the standard Gaussian cumulative distribution function. The GELU nonlinearity weights inputs by their percentil, rather than gates inputs by their sign as in ReLU ($x1_{x>0}$). Consequently the GELU can be thought of as a smother ReLU.
$$
\text{GELU}(x) = xP(X \le x) = x\Phi(x) = x \cdot \frac {1}{2}[1 + \text{erf}(x/\sqrt{2})]
$$
This can also be approximated with  $0.5x(1 + \tanh[\sqrt{2/\pi} (x + 0.044715x^3)])$ or $x\sigma(1.702x)$

#### QuickGELU

```python
def quick_gelu(x):
  return x * torch.sigmoid(1.702 * x)
```

### SwiGLU

used in Llama-2, PaLM

SwiGLU is an activation function which is a variant of GLU.
$$
\text{SwiGLU}(x, W, V, b, c, \beta) = \text{Swish}_{\beta}(xW+b) \otimes (xV+c)
$$




## Optimizers

### GD

$$
\theta = \theta - \alpha \nabla L(\theta; x, y)
$$

### SGD

$$
\theta = \theta - \alpha \nabla L(\theta; x^{(i:i+n)}, y^{(i:i+n)})
$$

### SGD+Momentum

When the magnitude of the gradient is too low, like around a local minimum or a saddle point, gradient descent gets stuck. With momentum, the parameters move by some fraction of the previous step along with the current gradient. 
$$
v_t = \lambda v_{t-1} + (1-\lambda)\nabla L(\theta_t-1) \\
\theta_t = \theta_{t-1} - \alpha v_t
$$

#### AdaGrad

AdaGrad adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters. For Adagrad, we keep a running total of the square of the gradients for each parameter. Denote this total for the $i$th parameter at timestep $t$ as $g_t^{i}$. The update step for $g$ is:
$$
g_t^i = g_{t-1}^i + \lparen \frac {\partial} {\partial \theta^{i}} L(\theta_{t-1})\rparen^2
$$
Then, when updating the parameters, we divide the learning rate by $\sqrt{g_t} + \epsilon$​. Intuitively, this will cause the parameters move less in directions where the magnitude of the gradient is large, and to move more in the directions where the gradient is low.
$$
\theta_t^{(i)} = \theta_{t-1}^{(i)} - \frac {\alpha} {\sqrt {g_t^{(i)}} + \epsilon} \nabla L(\theta_{t-1})
$$
The value of $g$ is positive and will monotonically increase over time. One problem is that the training will eventually stagnate as the learning rate goes to zero.

### RMSProp

RMSProp is similar to Adagrad, but seeks to solve the problem of the decaying learning rate. Instead of storing a sum of all of the past squared gradients, RMSProp keeps a exponentially decaying running average.

The running average is updated as follows:
$$
g_t^{(i)} = \rho g_{t-1}^{(i)} + (1-\rho)(\nabla L(\theta_{t-1}))^2
$$
Then, the parameter update is the same as for Adagrad. Dividing the learning rate by this exponential average will cause the learning to speed up in directions with lower gradients.
$$
\theta_{t}^{(i)} = \theta_{t-1}^{(i)} - \frac {\alpha} {\sqrt{g_{t}^{(i)}} + \epsilon} \nabla L(\theta_{t-1})
$$

###  Adam (Adaptive Moment Estimation) Optimizer

Adam is an iteration of previous algorithms which tries to estimate the first and second moments of the gradients.

The first moment is the expected value of the gradients, and the second moment is the expected value of the square of the gradients. Adam uses exponentially decaying averages to estimate these moments based on past gradients. $\beta_1$ and $\beta_2$ are the scaling factors for the exponential average, and the default values of $\beta_1 = 0.9 $ and $\beta_2 = 0.999$​
$$
m_t^{(i)} = \beta_1 m_{t-1}^{(i)} + (1 - \beta_1)(\nabla L(\theta_{t-1})) \\
v_{t}^{(i)} = \beta_2 v_{t-1}^{(i)} + (1 - \beta_2)(\nabla L(\theta_{t-1}))^2
$$
However, these estimates are biased towards 0, so Adam scales them as follows:
$$
\hat{m} = \frac {m} {1 - \beta_1^{t}} \\
\hat{v} = \frac {v} {1 - \beta_2^{t}}
$$
The parameter update for Adam is then:
$$
\theta_t^{(i)} = \theta_{t-1}^{(i)} - \frac {\alpha} {\sqrt{\hat{v}_{t}^{(i)}} + \epsilon} \hat{m}_{t}^{(i)}
$$


### AdamW

Adam with weight decaying, or L2 regularization.
$$
\theta_t^{(i)} = (1 - \alpha \lambda) \theta_{t-1}^{(i)} - \frac {\alpha} {\sqrt{\hat{v}_{t}^{(i)}} + \epsilon} \hat{m}_{t}^{(i)}
$$

## Regularization

### L2 regularization (Weight Decay)

A Linear regression model with L2 regularization is called a **Ridge Regression**. L2 regularization is popular in deeper neural nets, like for most transformer models, typically used together with the Adam optimizer. 

#### Gradient Implementation

$$
L_{L2}(\theta) = L(\theta) + \frac {\lambda} {2}\sum_{k=1}^K\theta_i^2 \\
\nabla_{\theta_i} L_{L2}(\theta_i) = \nabla_{\theta_i}L(\theta) + \lambda \theta_i \\
\theta_i = \theta_i - \alpha \nabla_{\theta_i} L(\theta) - \alpha \lambda \theta_i \\
\text{or  } \theta_i = (1- \alpha \lambda)\theta_i - \alpha \nabla_{\theta_i} L(\theta)
$$

### L1 regularization

A linear regression model with L1 regularization is called a **Lasso Regression**. One with both L2 and L1 regularizations is called an **Elastic Net**. L1 regularization shrinks some parameters towards zero (encourages sparsity), resulting in a model with practically less parameters.

#### Gradient Implementation

$$
L_{L1}(\theta) = L(\theta) + \lambda\sum_{k=1}^K|\theta_i| \\
\nabla_{\theta_i} L_{L2}(\theta_i) = \nabla_{\theta_i}L(\theta) + \lambda \text{sign}(\theta_i) \\
\theta_i = \theta_i - \alpha \nabla_{\theta_i} L(\theta) - \lambda \text{sign}(\theta_i)
$$

### Dropout

Another common regularization strategy: randomly set some fraction of the activations at each layer to zero. Dropout is typically used in deeper neural networks like in ResNet and Transformers, typically used after several convolutional layers, between dense layers, and when applying the attention weights to value vectors (in Transformer-based models).
$$
\hat {z_{i+1}} = \sigma_i(W_i^Tz_i + b_i) \\
(z_{i+1}) = \begin{cases}
(\hat{z_{i+1}})/(1-p) & \text{with probability} 1- p \\
0 & \text{with probability} p
\end{cases}
$$


```python
class Dropout(nn.Module):
  def __init__(self, p):
    self.p = p
  def forward(self, x):
    # during inference, Dropout behaves like a identity mapping
    if self.p != 0 or self.training:
			mask = torch.new_empty(x) # torch.new_ method returns a tensor with the same dtype and device as the argument tensor
      mask.bernoulli_(1 - self.p)
      mask.div_(1 - self.p)
			x = x * mask
    return x
```

### Stochastic Depth

Typically used together with residual connection, so during training, for some data point, they pass a stochastic amount of layers.

```python
class StochasticDepth(nn.Module):
  def __init__(self, p):
    self.p = p
  def forward(self, x):
    if self.p != 0 and self.training:
      batch_size = x.shape[0]
      size = [batch_size] + [1]*(x.dim()-1) # create a per-data-example mask
      mask = torch.empty(size,dtype=x.dtype, device=x.device)
      mask.bernoulli_(1 - self.p)
      mask.div_(1 - self.p)
      x = x * mask
    return x
```



## Normalization

### BatchNorm

```python
```



### LayerNorm

A vanilla feed-forward network might suffer from *internal covariate shift* issue, where a layer's input distribution changes as previous layers are updated. This could negatively affect the stability of parameter's gradients, delaying model converage.

To reduce this shift, LayerNorm normalizes the summed inputs so as to fix their mean and variances as follows:

```python
class LayerNorm(nn.Module):
  def __init__(self, hidden_size):
    self.bias = nn.Parameter(torch.zeros(hidden_size))
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.running_mean = 0
    self.running_std = 0
  def forward(self, x):
    x /= x.std(dim=0)
    x -= x.mean(dim=0)
    
```





### RMSNorm

The re-scaling invariance is the reason for success of LayerNorm, rather than re-centering invariance.

```python
class RMSNorm(nn.Module):
  def __init__(self, hidden_size, eps=1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
  
  def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return hidden_state
```

## Gradient Accumulation

### Implementation

```python
def train(model, optimizer, data_loader, gradient_accumulation_steps):
  model.train()
  for i, batch in enumerate(data_loader):
    inputs, labels = batch
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(labels, logits) / gradient_accumulation_steps
    loss.backward()
    if (i+1) % gradient_accumulation_step == 0:
      optimizer.step()
      optimizer.zero_grad()
```





## Loss Functions



## LoRA (Low Rank Adaptation)

### Rationale

Some work showed that the learned over-parameterized models in fact reside on a low intrinsic dimension. We hypothesized that the change in weights during model adaptation also has a low "intrinsic rank".

Compared to **parameter-efficient adaptation** method, which **inserts an adapter layer between existing layers** in a model, the LoRA finetuning **does not introduce latency**. Compared to prompt-tuning methods, like prefix tuning, the LoRA finetuning **does not increase the input sequence length**.

### Implementation

We initialize the matrix `A` with a random distribution with a standard deviation `1/sqrt(rank)`, which makes sure that magnitude of `A @ B` is 1.  We initialized `B` **with zeros**. The rationale here is that at the beginning of the training, before `A` and `B` are updated via backpropagation, the `LoRALayer` does not impact the original weights because $A B = \bar 0$.

```python
class LoRALayer(nn.Module):
  """
  params:
  	alpha: the scaling factor determining the magnitude of the changes introduced by the LoRA layer to the model's existing weights
  """
  def __init__(self, in_dim, out_dim, rank, alpha):
    std_dev = 1 / torch.sqrt(torch.tensor(rank)).float()
    self.A = torch.nn.Paremeter(torch.randn(in_dim, rank) * std_dev)
    self.B = torch.nn.Paremeter(torch.zeros(rank, out_dim))
    self.alpha = alpha
	def forward(self, x):
    x = self.alpha * (x @ self.A @ self.B)
    return x
```

And then we should modify the forward method of each layer that is to be updated by LoRA. An easy to implement this is to replace each layer with a wrapper `LayerWithLoRA` that combines the original layer and a `LoRALayer`

```python
class LinearWithLoRA(torch.nn.Module):
  def __init__(self, linear, rank, alpha):
    super().__init__()
    self.linear = linear
    self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
  def forward(self, x):
    return self.linear(x) + self.lora(x)
  
  def merge_weights(self):
    # the weight is transposed in nn.Linear
    # so the shape of weight is [out_dim, in_dim]
    self.linear.weight.data += self.lora.alpha * (self.lora.A @ self.lora.B).transpose()
    
```

Then we can use `functools.partial` to wrap layers we would like to fine-tune using LoRA.

```python
from functools import partial

# default hyperparameter choices
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False

assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
detach_lora = lambda lora_layer: lora_layer.linear

# freeze all parameters in the model
for p in model.parameters():
  p.requires_grad = False

for layer in model.transformer.layer:
    if lora_query:
        layer.attention.q_lin = assign_lora(layer.attention.q_lin)
    if lora_key:
        layer.attention.k_lin = assign_lora(layer.attention.k_lin)
    if lora_value:
        layer.attention.v_lin = assign_lora(layer.attention.v_lin)
    if lora_projection:
        layer.attention.out_lin = assign_lora(layer.attention.out_lin)
    if lora_mlp:
        layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
        layer.ffn.lin2 = assign_lora(layer.ffn.lin2)
if lora_head:
    model.pre_classifier = assign_lora(model.pre_classifier)
    model.classifier = assign_lora(model.classifier)
    
# training
for _ in epochs:
  train_one_epoch(model, dataloader)
  
# finalize the weight update and detac lora layers
for layer in model.transformer.layer:
		if lora_query:
        layer.attention.q_lin = detach_lora(layer.attention.q_lin)
    if lora_key:
        layer.attention.k_lin = detach_lora(layer.attention.k_lin)
    if lora_value:
        layer.attention.v_lin = detach_lora(layer.attention.v_lin)
    if lora_projection:
        layer.attention.out_lin = detach_lora(layer.attention.out_lin)
    if lora_mlp:
        layer.ffn.lin1 = detach_lora(layer.ffn.lin1)
        layer.ffn.lin2 = detach_lora(layer.ffn.lin2)
if lora_head:
    model.pre_classifier = detach_lora(model.pre_classifier)
    model.classifier = detach_lora(model.classifier)
```

