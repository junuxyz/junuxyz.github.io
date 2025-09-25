+++
title = "Shaped Transformer"
date = 2025-09-15T07:39:43+09:00
draft = false
categories = ['ML']
tags = ['Transformer']
+++

## 0. Understanding Transformer

**How can one learn Transformer?**

The Transformer Architecture (introduced in the paper _[Attention is All You Need](https://arxiv.org/abs/1706.03762)_) is one of the most successful models in deep learning and the backbone of what made the “ChatGPT moment” possible. Because of its importance and impact, there are already many high-quality explanations of [what the model is](https://jalammar.github.io/illustrated-transformer/), [how it works](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/), and even [annotated code implementation of it](https://nlp.seas.harvard.edu/annotated-transformer/). These days, most developers don’t need to implement Transformers from scratch because libraries like [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) provide easy-to-use classes and methods. Yes, there are plenty of things to build on top of the architecture! Still, I think it is worth having a great understanding of Transformer Model, beyond intuitive, abstract level. In fact one of the best way to learn Transformer, as [Feynman said](https://www.goodreads.com/quotes/7306651-what-i-cannot-build-i-do-not-understand), is to build one yourself from scratch to really understand and appreciate all the underlying techniques and modules that form the base of the ChatGPT era.


**How is this different from other content?**

I do strongly recommend reading other resources as well. I believe each sources has different layers of abstraction (or depth of explanation). The [paper itself](https://arxiv.org/abs/1706.03762) is fairly straightforward but not chronologically ordered, so it can be hard to follow in details. _[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)_ is beginner-friendly, abstracting away many implementation details and excels at explaining the overall big picture. On the other hand, _[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)_ is very deep, building the entire architecture end to end in PyTorch. But since it follows the paper’s order (which isn't chronological) and leaves out some explanations, readers who only have an abstract understanding of the model may feel overwhelmed or questionable.

Also note that Transformer is not a single monolithic block—it’s made up of many modularized layers (tokenization, positional encoding, encoder-decoder model, self-attention, cross-attention, etc.). Unless you already have a solid background in deep learning and NLP, it’s hard to fully understand all the pieces in one go. You’ll often need additional resources, and repeated exposure, to get comfortable with it.

While there are many great explanations of the mathematics and abstract concepts, I think the end-to-end **shape changes** and detailed explanation of code implementation are often missing. This blog post specifically aims to enhance the reader’s intuition about what the input actually looks like in real code, how it gets transformed step by step, and how it eventually can successfully predict the “next token”.

Hopefully this helps you form a more concrete understanding of the architecture and makes the code easier to implement :)

<br>

## 1. Commonly Used Parameters

Before we talk about shape transformation, it is helpful to understand the names of the parameter/notations. It will help the code readability. If you are famaliar with the paper and the parameters used, feel free to skip this section.

### $N$ , $b$, or `nbatches`

The Paper use the expression $N$ but in code, it is expressed as `nbatches`.

The reason why you may be confused about `nbatches` in code implementation from Annotated Transformer is because most explanation (including the original paper) omit about it.


The most representative image of Transformer is usually
1. One head from one batch or
2. Multi-Head from one batch

but they don't explicitly tell there are `nbatches` batches processed parallely for each batch.

The reality is, $N$ sentences are put into batch and passed as input. But this doesn’t change or introduce anything new to the original architecture we know. Still it's worth noting that `nbatches` mean the number of sentences being processed per batch. It will appear in the code several times.

Ex. let’s say $N=3$. That means we have $3$ sentences per batch.


### $S$ or `n_seq`

`n_seq` means the number of tokens in one sentence. Since Transformer utilizes parallel processing, we need to pre-define (statically) the length of the sentence. Usually we define it based on the longest sentence.

For example,

```
I love you            --3
I am in love with you --6
I love you, too       --5
```

In this case, since the longest sentence in the batch is $6$, we can set `n_seq = 6`.
For the sentences that have less tokens than 6 will be filled with mask. We will see how mask(padding) is implemented later in this post.

### $d_{\text{model}}$ or `d_model`

`d_model` is the dimensionality of the **residual stream**, i.e., the vector size that represents each token throughout the Transformer.

All major components, including Embedding, Attention, and Feed-Forward layers, produce outputs of shape `(N, S, d_model)`. This uniform dimension ensures that residual connections (`x = x + sublayer(x)`) can be applied seamlessly across all sublayers.

In the original paper, `d_model` was set to 512.

### $vocab$ or `vocab`

`vocab` is number of all token (or token ID). It depends on how you tokenize it.

I think prior resources didn’t explain about the exact input of Transformer architecture clearly but I think it’s worth noting.

First, even before the transformer process begins, there is a thing called Tokenizer which is independent from the Transformer Architecture. The tokenizer splits raw sentences into seqeunce of tokens.

For example if the raw sentence input was `I love you`, Tokenizer would divide it into tokens,

```
["I", "love", "you"]
```

and using the $vocab$ dictionary, we map the tokens with its correspoining token id (one-on-one match)

```
[0, 1, 2]
```

Now **this (sequences of token id)** is the input of the Transformer Architecture. 

Then you might ask **what's the input of Transformer then?**

The very first step of Transformer is Embedding and this is done by selecting the row from `W_emb` using the token ID as a key: `W_emb[token ID]`

```
token_id = 1 ("love")
→ W_emb[1] = [0.13, -0.25, 0.91, ..., 0.07] 
```

Since the vector representation of token has size of $d_\text{model}$ (as mentioned above) The shape of $W_\text{emb}$ will be $(vocab, d_{\text{model}})$.

tldr:
- `vocab` means number of all possible tokens, which depends on the Tokenizer.
- After the token is converted into token id, the sequence of token ids become the actual “input” of the Transformer Architecture (The most left, below from the Transformer Architecture image)

### Parameters specifically used During Attention Calculation

Below are the parameters only seen in Attention Calculation (Self-Attention and Cross-Attention)

### $H$, $h$, or `h`

`h` is a hyperparameter which means the number of head of Multi-Head Attention.
In the original paper, researchers set it as `h = 8`.

### $d_k$ or `d_k`

`d_k` is the vector size of key($K$) representation of each token.

We will look into more detail about how shape transforms during Multi-Head Attention in the upcoming section, but to shortly address, $Q$(query) and $K$(key) are matrix multiplied to get the Attention Score. Therefore `d_q` must be the same as `d_k`.

In the paper `d_k = d_model // h` which is $64$ ($=512/8$). Most people think `d_k` must be `d_model // h` but this is just a design choice and totally depends on the developer. I will explain cases where this is not always true, while it's still efficient to use `d_k = d_model // h`.

### $d_v$ or `d_v`

`d_v` is the dimension of the value($V$) vector for each token.

After the attention weights are calculated, they are multiplied by the Value matrix $V$. This process yields a new set of vectors, each with dimension `d_v`, that now holds the contextual information from the sequence. This output is then used to help predict the next token. 
(Don't worry if this sounds too compact. I will explain it in more detail in the next section!)

In the original "Attention Is All You Need" paper, the authors set `d_v = d_k = d_q`. However, while `d_k` must equal `d_q`, it's not required for d_v to be the same size. This is simply another design choice. I will also explain later in this post when `d_v != d_k` is acceptable.


<br>

## 2. How Shape Changes, end to end (w/ Code Examples)

When we encounter the code implementation of Transformer Architecture, all kinds of `view()`, `transpose()`, `reshape()` methods are frequently used, hindering what are being changed and what it implies. After all, if you understand there is a general order of the code and the shape form all has its meanings, code readability enhances significantly.

Before we start, a simple but effective tip is to remember that most calculations 
(token embedding, self-attention, feed-forward, etc.) are applied **per token**.
In practice, the input shape is `(nbatches, n_seq, …)`, which means each sequence has `n_seq` tokens and each batch has `nbatches` sentences. Almost all operations are performed independently on each of these tokens (in parallel), except for the masked(or, *padded*) ones. So you can think of it as running the same function `nbatches × n_seq` times in parallel.

Now, let's explore the journey from the very first embedding to the last output (next token) and see how the shape changes and what they all mean. (In most paper or images, they often omit `nbatches` or `h` for clarity but I will explain including all the parameters). Then we will see how code is actually written to match these shapes. We won't cover all code, just the shape transformation parts, for simplicity. Also, to give you a clear intuition of how everything is working, I will use the three following sentences I used above. Code can be found in ... :


### Tokenizing

**shape: `(nbatches, n_seq)`**

This is a step before the transformer architecture even starts. It's a process to convert raw sentences into sequence of tokens. For example, based on [GPT-4o & GPT-4o mini tokenizer](https://platform.openai.com/tokenizer) provided by OpenAI,

```Plain Text
I love you            
I am in love with you
I love you, too
```

are converted into discrete tokens:

![[shaped-trasnformer-example-1.png]]

{{< note >}}
The only reason we use the GPT-4o tokenizer here is since it's the most convenient tokenizer available on the web. However all the rest of the concepts and parameters (e.g. size of $vocab$) will be based on the original paper!
{{< /note >}}


We use padding tokens(`PAD`) to keep the length of all `n_seq` same (which is crucial for matrix calculation). If we set `n_seq` to 6, the padding will fill as below:

![[shaped-trasnformer-example-2.png]]

These tokens converted to token ids will be

![[shaped-trasnformer-example-3.png]]

This will be our exact starting point (input) for Transformer Architecture.


{{< qa question="Why is the shape `(nbatches, n_seq)` sometimes described as `(nbatches, n_seq, vocab)` if each token ID is just a scalar value?" >}}

In the original paper, the authors simply state that they use _learned embeddings_ to map token IDs to vectors of dimension $d_{model}$, without mentioning one-hot explicitly. 

Mathematically, however, you can think of each token as a one-hot vector, so that the embedding operation becomes


$$x_{\text{emb}} = \text{onehot}(token_{id}) \times W_{emb}.$$

This is convenient because it lets us express the embedding as a standard matrix multiplication.

For example, the sentence `"I love you"` with token IDs `[40, 3047, 481, 0, 0, 0]` would look like:
```mathematica
O[0, 0, :] = [0, 0, ..., 1(at 40), ..., 0]    # "I"
O[0, 1, :] = [0, 0, ..., 1(at 3047), ..., 0]  # "love"
O[0, 2, :] = [0, 0, ..., 1(at 481), ..., 0]   # "you"
O[0, 3, :] = [1(at 0), 0, 0, ..., 0]          # [PAD]
O[0, 4, :] = [1(at 0), 0, 0, ..., 0]          # [PAD]
O[0, 5, :] = [1(at 0), 0, 0, ..., 0]          # [PAD]
```


Of course, this representation is very inefficient in practice (huge memory cost). 

So in real implementations (e.g. PyTorch), we directly use the `(nbatches, n_seq)` token ID tensor to _index_ into `W_emb` and fetch the corresponding rows.

**In practice:** think of `vocab` as the _number of unique token IDs (vocabulary size)_, not as an actual one-hot dimension in the input.
{{< /qa >}}


### Token Embedding

**input shape: `(nbatches, n_seq)`**

Now the Transformer Architecture starts. 
First thing we do is we embed all token in sequences within a batch.

Embedding matrix `W_emb` shape is `(vocab, d_model)`. In the original paper `d_model` is set to 512.

Input(`(nbatches, n_seq)`) will use `W_emb` as a lookup table and replace the token id into `d_model` dimension, which makes the scalar token id into `d_model` dimension vector for each token.

After embedding, we scale the elements in d_model by multiplying it with $\sqrt{d_\text{model}} = \sqrt{512}$. Original paper doesn't explain the reason clearly but based on this, it seems they are multiplying to keep/strengthen the token embedding information even after Positional Embedding is added. Since it's just scalar multiplication, this procedure doesn't change the shape.

output shape: `(nbatches, n_seq, d_model)`


Back to our example, the sentence "I love you"(`40 ,3047, 481, 0, 0, 0`) will extract `W_emb[40]`, `W_emb[3047]`, `W_emb[481]`, and `W_emb[0]`. 



|  token  | token id |                         d_model = 512                         |
| :-----: | :------: | :-----------------------------------------------------------: |
| `[PAD]` |    0     |   `[ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, ... ]`   |
|   `I`   |    40    | `[ 0.12, -0.07, 0.91, -0.04, 0.26, 0.03, -0.18, 0.55, ... ]`  |
| `love`  |   3047   | `[ 0.33, 0.25, -0.48, 0.18, -0.09, 0.41, 0.07, -0.22, ... ]`  |
|  `you`  |   481    |  `[-0.55, 0.19, 0.07, 0.92, -0.14, 0.08, 0.36, -0.02, ... ]`  |
|  `am`   |   939    |  `[ 0.41, 0.05, 0.36, -0.33, 0.22, -0.17, 0.10, 0.04, ... ]`  |
|  `in`   |   306    | `[ 0.29, -0.15, -0.04, 0.22, -0.11, 0.09, -0.27, 0.30, ... ]` |
| `with`  |   483    |  `[-0.21, 0.56, 0.02, 0.44, 0.05, -0.08, 0.12, -0.19, ... ]`  |
|   `,`   |    11    | `[ 0.77, -0.10, -0.13, -0.26, 0.15, 0.06, -0.05, 0.02, ... ]` |
|  `too`  |   3101   | `[ 0.04, 0.88, -0.29, 0.13, -0.06, 0.21, -0.12, 0.44, ... ]`  |

![[shaped-transformer-example-4.png]]

After replacing based on the W_emb lookup table, "I love you" would be

```mathematica
E[0, 0, :] = [ 0.12, -0.07,  0.91, -0.04,  0.26,  0.03, -0.18,  0.55, ... ]   # "I"    (id=40)
E[0, 1, :] = [ 0.33,  0.25, -0.48,  0.18, -0.09,  0.41,  0.07, -0.22, ... ]   # "love" (id=3047)
E[0, 2, :] = [-0.55,  0.19,  0.07,  0.92, -0.14,  0.08,  0.36, -0.02, ... ]   # "you"  (id=481)
E[0, 3, :] = [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, ... ]   # [PAD]  (id=0)
E[0, 4, :] = [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, ... ]   # [PAD]  (id=0)
E[0, 5, :] = [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, ... ]   # [PAD]  (id=0)
```

Of course other two sentences in the batch, "I am in love with you" and "I love you, too" also will go through the same process.

### Positional Embedding

We won't go into detail about positional embedding in this post, since itself is a post-worth concept. I recommend [this post](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) for better understanding of Positional Embedding. Since our main focus here is to see how the shapes change, I will explain mainly on that point of view.

**input shape: `(nbatches, n_seq, d_model)`**

To briefly explain, Positional Embedding is adding same(`d_model`)-sized vector to each embedded token to preserve the positional information. Unlike other sequence neural networks such as LSTM or RNN, Transformer calculates all tokens parallely. Therefore, if it doesn't have positional information added, it wouldn't know the order of the token. While using $\sin$ and $\cos$ functions are not the only way to preserve positional information, it is known to be a helpful way to do that.
We add the positional information of `d_model` to each embedded tokens and since they have the same shape, the shape does not change.

**output shape: `(nbatches, n_seq, d_model)`**


Back to our example, "I love you" has three positions.

```mathematica
PE[0, :] = [ 0.00,  1.00,  0.00,  1.00,  0.00,  1.00,  0.00,  1.00, ... ]   # pos=0
PE[1, :] = [ 0.84,  0.54,  0.84,  0.54,  0.84,  0.54,  0.84,  0.54, ... ]   # pos=1
PE[2, :] = [ 0.91, -0.42,  0.91, -0.42,  0.91, -0.42,  0.91, -0.42, ... ]   # pos=2
PE[3, :] = [ 0.14, -0.99, 0.14, -0.99, 0.14, -0.99, 0.14, -0.99, ... ] # pos=3 
PE[4, :] = [-0.76, -0.65, -0.76, -0.65, -0.76, -0.65, -0.76, -0.65, ... ] # pos=4
PE[5, :] = [-0.96, 0.28, -0.96, 0.28, -0.96, 0.28, -0.96, 0.28, ... ] # pos=5
```

We add the PE with E for every token:
```mathematica
E+PE[0, 0, :] = [ 0.12+0.00, -0.07+1.00,  0.91+0.00, -0.04+1.00, ... ]   # "I"
           = [ 0.12, 0.93, 0.91, 0.96, 0.26, 1.03, -0.18, 1.55, ... ]

E+PE[0, 1, :] = [ 0.33+0.84, 0.25+0.54, -0.48+0.84, 0.18+0.54, ... ]     # "love"
           = [ 1.17, 0.79, 0.36, 0.72, 0.75, 0.95, 0.91, 0.32, ... ]

E+PE[0, 2, :] = [-0.55+0.91, 0.19-0.42, 0.07+0.91, 0.92-0.42, ... ]     # "you"
           = [ 0.36, -0.23, 0.98, 0.50, 0.77, -0.34, 1.27, -0.44, ... ]

E+PE[0, 3, :] = [ 0.00+0.14, 0.00-0.99, 0.00+0.14, 0.00-0.99, ... ]     # [PAD] at pos=3
           = [ 0.14, -0.99, 0.14, -0.99, 0.14, -0.99, 0.14, -0.99, ... ]

E+PE[0, 4, :] = [ 0.00-0.76, 0.00-0.65, 0.00-0.76, 0.00-0.65, ... ]     # [PAD] at pos=4
           = [-0.76, -0.65, -0.76, -0.65, -0.76, -0.65, -0.76, -0.65, ... ]

E+PE[0, 5, :] = [ 0.00-0.96, 0.00+0.28, 0.00-0.96, 0.00+0.28, ... ]     # [PAD] at pos=5
```

	We can see that even though the padding tokens have no semantic meaning, they still receive a unique positional signal before being processed by the Transformer layers. 

As always, **remember** this process is done by all sequences in the batch! (which means this process is parallelly done in the sentences "I am in love with you" and "I love you, too" as well!)


### Encoder

Now we move to the Encoder Layer. Code implementation of Encoder (and Decoder) Layer in code seems like a matryoshka doll (module inside a module inside a moduel...) but I will try to explain the big picture as clearly as possible.

In the blueprint, **Encoder** is consisted of $N$ Layers of **Encoder Layer**. (Note that Encoder and Encoder Layer are NOT the same! Also note that $N$ here and `nbatches` are totally different, independent concepts!) In the original paper $N=6$ which means we go over the Encoder layer 6 times in a single process.

Each Encoder Layer has two **Sublayers**: **Self-Attention Sublayer** first and then **Feed Forward Network($FFN$) Sublayer**. After each sublayers are passed, they are connected to a residual stream and then layer-wise normalized(LayerNorm).

**Sublayers** are the most fundamental building blocks in Encoder-Decoder structure of Transformer Architecture. The two types of sublayers are already mentioned above.

Now that we went over the modules inside Encoder in a brief top-down approach, we will first go over the shape/dimension changes in Self-Attention Sublayer, then Feed Forward Network Sublayer and will expand further.


### Self-Attention (Multi-Head Attention)

Now this module is one of the most complicated part in Transformer so we will look in detail, step by step. The goal is to get a clear sense of how shape changes during each process. We are going to divide the Self-Attention process into 8 minor steps. Try to follow through!


![[shaped-attention-attention.png]]

Recap: Each token are `d_model` sized vectors, where token embedded and positional embedded vectors are added.

**input shape: `(nbatches, n_seq, d_model)`**

When the input enters an Encoder Layer, its first destination is the Self-Attention sublayer. To understand what happens here, we need to talk about **Multi-Head Attention**.

Though this post isn't a deep theoretical dive, the core idea is simple: instead of calculating attention just once, we use multiple "attention heads." Think of it like having several experts look at the same sentence; each expert (or head) can focus on different relationships between words. Each head has its own set of learned weights (`W_Q`, `W_K`, `W_V`) to find these different relationships. In the original paper, they use 8 heads (`h=8`).

Mathetmatically we represent Attention calculation as 

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

Let's break down how the actual tensor shapes transform step-by-step.

{{< note >}}
From now, we will call the input as `x`.
{{< /note >}}


**1. Project to $Q, K, V$** `(nbatches, n_seq, d_model)` -> `(nbatches, n_seq, d_model)`
We first need to get the $Q, K, V$ matrices. This is done by multiplying our input `x` with three learned weight matrices: `W_q`, `W_k`, and `W_v`. Each of the weight matrices have size of shape `(d_model, d_model)`.

Each $Q, K, V$ becomes shape `(nbatches, n_seq, d_model)`
- `Q = x @ W_q = (nbatches, n_seq, d_model)`
- `K = x @ W_k = (nbatches, n_seq, d_model)`
- `V = x @ W_v = (nbatches, n_seq, d_model)`

{{< note type="tip" >}}
**A Note on Parameters:** We aren't training `Q`, `K`, and `V` directly. 
The actual parameters we train are the weights: `W_Q`, `W_K`, and `W_V`.
{{< /note >}}

**2. Splitting into Heads** `(nbatches, n_seq, d_model)` -> `(nbatches, n_seq, h, d_k)`
After projection, we split `d_model` into `h` seperate heads. Since in the paper, d_k is defined as `d_k = d_model / h`, we can divide the last dimensions into `h` and `d_k` and view the shape as `(nbatches, n_seq, h, d_k)`.

In code, it will be implementing as follow:


```python
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
	...
	self.d_k = d_model // h
	self.h = h
	self.linears = clones(nn.Linear(d_model, d_model), 4)
	...

	def forward(self, query, key, value, mask=None):
	...
	query, key, value = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for lin, x in zip(self.linears, (query, key, value))]
	...
```

Let's break down the last line.

For conveninece we make four Linear Layer of shape `(d_model, d_model)` for `W_q, W_k, W_v, W_o` and save it in ModuleList.

{{< note >}}
To be accurate it's `(d_model, h, d_k)` but for sake of convenience, since `d_model = h * d_k`, we write as `(d_model, d_model)` for sake of convenience.
{{< /note >}}

We `zip` it with tuple `(query, key, value)`. Since we don't need q,k,v for `W_o`, it is intended to zip with only three tuples.
Since `query, key, value` are all shape of `(nbatches, n_seq, d_model)`, we pass it through `lin(x)`. since the Linear Layer itself has the same input and output size, the output of query, key, value after the Linear Layer have the same shape `(nbatches, n_seq, d_model)`. However we transpose it to `(nbatches, h, n_seq, d_k)` to make matrix multiplication available for parallel processing of each heads.





**3. Transpose for Attention Calculation**
To perform the attention calculation ($QK^T$) across all heads at once, we need `n_seq` and `d_k` to be the last two dimensions. So we transpose the shape $Q$ into `(nbatches, h, n_seq, d_k)`. Since $K^T$ is transposed version of $K$, its shape is `(nbatches, h, d_k, n_seq)`.

**4. Calculate Attention Scores**
We calculate $QK^T$ and the output shape becomes `(nbatches, h , n_seq, n_seq)` due to matrix mulitplicatin. 
After calculating, as you see in the mathematical definition, we scale it by $\frac{1}{\sqrt{d_k}}$ to make it `n_seq`-agnostic.
Mathematically this process is 

**6. Mask Padding Tokens**
Padding tokens are just placeholders to match all the sentence to length of `n_seq` for the sake of matrix calculation. Therefore we need to exclude them on applying to $V$. Therefore we mask the padding tokens in the key. We don't need to mask the query matrix since it matrix multiplies with masked key tokens and automatically gets eliminated.

**7. Apply Softmax** 
We apply a softmax function to the scores, which turns them into positive values that sum to 1. This converts the scores into attention **weights**. The shape remains `(nbatches, h, n_seq, n_seq)`.

**8. Apply Attention to V** Now we multiply our attention weights by the Value matrix `V`. This enhances the representation of each token by incorporating information from the other tokens it's paying attention to.
Attention weight shape: `(nbatches, h, n_seq, n_seq)`
`V` shape: `(nbatches, h, n_seq, d_v)` (where `d_v = d_k` in the original paper)
Output = `weights @ V` and its shape becomes `(nbatches, h, n_seq, d_v)`

**9. Concatenate Heads**
The parallel processing for each head is done. Now we want to merge our `h` heads back into a single tensor. We reverse the transpose from step 3.
so the shape changes from `(nbatches, h, n_seq, d_v)` to `(nbatches, n_seq, h, d_v)`.

**10. Reshape to Final Output** 
Finally, we flatten the `h` and `d_v` dimensions back into the original `d_model` dimension. This is the `reshape()` or `view()` call(in code) that concludes the process.
Shape changes from `(nbatches, n_seq, h, d_v)` to `(nbatches, n_seq, h * d_v)`.
Since `h * d_v = d_model`, our final output shape is `(nbatches, n_seq, d_model)`, which is exactly what we started with! This allows us to pass it to the next layer in the network.

This output is then passed through a final linear layer, `W_O`, which also doesn't change the shape.

final output shape: `(nbatches, n_seq, d_model)`

{{< note >}} 
- In practice, each attention head receives the **same full input** of shape `(nbatches, n_seq, d_model)`. The difference between heads comes entirely from their learned projection weights $(W_Q, W_K, W_V)$. Heads do not split the input; instead, they naturally learn to focus on different subspaces of the same representation.

Below are two common parts most people are wrong about, or didn't think of.

- Does `d_k` have to be `d_model / h`?
	No, but it's a very practical choice to make.
	The main reason is to ensure the output of the attention block matches the shape as the input. This is crucial for the residual connection(`x = x + MHA(x)`). 
	When `d_k = d_model / h`, After we concatenate the heads, the final dimension is `h * d_k = h * (d_model / h) = d_model`. The output shape `(..., d_model)` perfectly matches the input shape, so the residual connection works seamlessly. When `d_k` is different from `d_model / h`, The concatenated dimension becomes `h * d_k`, which is not equal to `d_model`. This creates a shape mismatch. To fix this, the final linear layer (`W_O`) must act as a projection, mapping the shape from `h * d_k` back to `d_model`. So it's just a practical and rational design choice to keep the dimension consistent.
- Does `d_k` have to match `d_v`?
	**No**, not at all.
	While the query and key dimensions (`d_q` and `d_k`) must be identical for the dot product to work, the value dimension (`d_v`) can be any size you want.
	The output dimension after applying attention and concatenating the heads is always `h * d_v`. The final linear layer, `W_O`, is responsible for projecting this `h * d_v` dimension back to `d_model` to ensure the residual connection works.
	In fact, the `W_O` layer makes the choice of `d_v` very flexible. Its job is to take whatever the heads output (`h * d_v`) and reshape it into the `d_model` dimension that the rest of the network expects. Setting `d_k = d_v` is just a common simplification.

To summarize both questions above, if we just make sure `d_k = d_q` (for attention calculation) and shape of `W_O` as `(h * d_v, d_model)`, `d_k` and `d_v` can be whatever integer it can be. However by matching `d_k` and `d_v` based on `d_model` makes residual connection seamless and computationally efficient.
{{< /note >}}

Back to our example, "I love you" sentence

### Feed Forward Network ($FFN$)

input shape: `(nbatches, n_seq, `
We use $FFN$ to add non-linearity to learn and represent in higher dimension. 


## 3. How Shape changes in code

Last part, I will show how the shape changes is implied in actual code (in Annotated Transformer, specifically). Since we have already went through the shape change intuition in section 2, it will be much easier to follow :)


## Conclusion

We have went through the common notations used in transformer, and used specific example to go through "one training phase" of Transformer Architecture. Finally we saw how shape changes in actual code (why all those `reshape` and `transpose` are used). Hope this post helps you gain insight and intuition of the bigger picture.
