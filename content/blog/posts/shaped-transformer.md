+++
title = "Shaped Transformer"
date = 2025-09-15T07:39:43+09:00
draft = false
categories = ['ML']
tags = ['Transformer']
+++

## 0. Understanding Transformer

**How can one learn Transformer?**
The Transformer Architecture (introduced in the paper _Attention is All You Need_) is one of the most successful models in deep learning and the backbone of what made the “ChatGPT moment” possible. Because of its importance and impact, there are already many high-quality explanations of what the model is, how it works, and even annotated code implementations. These days, most developers don’t need to implement Transformers from scratch because libraries like HuggingFace provide easy-to-use classes and methods. There are plenty of things to build on top of the architecture! Still, I think it is worth implementing a Transformer from scratch at least once, to really understand and appreciate the techniques that form the base of the ChatGPT era.


**How is this different from other content?**
Before I start, I strongly recommend reading other resources as well. Each has a different abstraction layer (or depth of explanation). The paper itself is fairly straightforward but not chronologically ordered, so it can be hard to follow as a blueprint. _The Illustrated Transformer_ is very beginner-friendly, abstracting away many implementation details and explaining the overall big picture. On the other hand, _The Annotated Transformer_ is very deep, building the entire architecture end to end in PyTorch. But since it follows the paper’s order (which is not consistent) and leaves out some explanations, readers who only have an abstract understanding of the model may feel intimidated or overwhelmed.

Another challenge is that the Transformer is not a single monolithic block—it’s made up of many modularized layers (tokenization, positional encoding, self-attention, cross-attention, etc.). Unless you already have a solid background in deep learning and NLP, it’s hard to fully understand all the pieces in one go. You’ll often need additional resources, and repeated exposure, to get comfortable with it.

While there are many great explanations of the mathematics and abstract concepts, I think the end-to-end **shape changes** are often missing. This blog post specifically aims to enhance the reader’s intuition about what the input actually looks like in real code, how it gets transformed step by step, and how it eventually becomes the “next token.” 

Hopefully this helps you form a more concrete understanding of the architecture and makes the code easier to implement :)


## 1. Commonly Used Parameters

Before we talk about shape transformation, it is helpful to understand the parameter and notation names. It will help the code readability.

### $N$ , $b$, or `nbatches`

The Paper use the expression $N$ but in code, it is expressed as `nbatches`.

The reason why you may be confused about `nbatches` is because most explanation (of course the original paper) omit about it.

The most representative image of Transformer is usually

1. One head from one batch or
2. Multi-Head from one batch

but they omit there is actually every layers are processed in batches.

The reality is, $N$ sentences are put into batch and passed as input. But this doesn’t introduce anything new to the original architecture (all sentences in batch are just parallely processed) but it’s still worth noting that `nbatches` mean that because it’s the default parameter in all shapes in the Transformer.

### $S$ or `n_seq`

`n_seq` means the number of tokens in one sentence. Since Transformer utilizes parallel processing, we need to pre-define the length of it. Usually we define it based on the longest sentence. For example let’s say $N=3$. Then we have 3 sentences per batch.

```
I love you            --3
I am in love with you --6
I love you, too       --5
```

In this case, since the longest sentence in the batch is 6, we can set `n_seq=6` . For the sentences that have less tokens than 6 will be filled with mask. We will see how mask is implemented later in this post.

### $d_{\text{model}}$ or `d_model`

`d_model` is a hyperparameter which is the size of the vector for all residual stream.

Transformer typically express token information as `d_model` . In all sublayers both in encoder and decoder, residual connection (`x = x + sublayer(x)`) is used. Therefore it is helpful to match the dimension space for the sake of that. The output of Embedding Module, Attention Block, and FFN are all `(N,S,d_model)` shaped. So basically, `d_model` is set to unify the dimension for residual connection.

In the paper, `d_model=512`

### $vocab$

`vocab` is number of all token (or token ID). It depends on how you tokenize it.

I think prior resources didn’t explain about the exact input of Transformer architecture clearly but I think it’s worth noting.

First, even before the transformer process begins, there is a thing called Tokenizer which is independent from the Transformer Architecture. The tokenizer splits raw sentences into seqeunce of tokens.

For example if the raw sentence input was “I love you”,

```
I love you
```

Tokenizer divides it into tokens,

```
["I", "love", "you"]
```

and using the $vocab$ dictionary, we map the tokens with its correspoining token id (one-on-one match)

```
[0, 1, 2]
```

Now **this (sequences of token id)** is the input of the Transformer Architecture. The first step of Transformer Architecture, is Embedding and this is done by selecting the row from W_emb using the token ID as a key: `W_emb[token ID]`

```
token_id = 1 ("love")
→ W_emb[1] = [0.13, -0.25, 0.91, ..., 0.07] 
```

Since the vector representation of token has size of $d_\text{model}$ (as mentioned above) The shape of $W_\text{emb}$ will be $(vocab, d_{\text{model}})$.

tldr:
`vocab` means number of all possible tokens, which depends on the Tokenizer.
After the token is converted into token id, that is the actual “input” of the Transformer Architecture (The most left, below from the Transformer Architecture image)

### Parameters specifically used During Attention Calculation

### $H$, $h$, or `h`

`h` is a hyperparameter that means the number of head of Multi-Head Attention.

In the original paper, researchers set it to `h=8`.

### $d_k$ or `d_k`

`d_k` is the vector size of key($K$) representation of each token.

We will look into more detail about how shape transforms in Multi-Head Attention Phase, but to shortly address, $Q$(query) and $K$(key) are matrix multiplied to get the Attention weight. Therefore `d_q == d_k`.

In the paper `d_k = d_model / h` which is $64$ ($=512/8$). Most people think `d_k` must be `d_model / h` but this is just a design choice and totally depends on the developer. I will show cases in the next section where `d_k` is NOT `d_model / h` and why it’s still efficient to use `d_k = d_model / h`.

### $d_v$ or `d_v`

`d_k` is the vectore size of value($V$) representation of each token.

After Attention weight is calculated, it is applied to `d_v` to get the value with all the context(implied in attention weight) which is used to predict the next token.

In the paper, it used `d_v = d_k = d_q` and while `d_k` must be `d_q`, `d_v` does NOT need to be `d_q` at all! Just another design choice made by the authors. I will show the case where `d_v != d_k` later in this post, as well.


## 2. How Shape changes, end to end

When we encounter the code implementation of Transformer Architecture, all kinds of `view()`, `transpose()`, `reshape()` methods are frequently used, hindering what are being changed and what it implies. After all, if you understand there is a general order of the code and the shape form all has its meanings, code readability enhances significantly.

Before we start, a simple but effective tip is to remember that most calculations 
(token embedding, self-attention, feed-forward, etc.) are applied **per token**.
In practice, the input shape is `(nbatches, n_seq, …)`, which means each sequence has `n_seq` tokens and each batch has `nbatches` sentences. Almost all operations are performed independently on each of these tokens (in parallel), except for the masked(or, *padded*) ones. So you can think of it as running the same function `nbatches × n_seq` times in parallel.

Now, let's explore the journey from the very first embedding to the last output (next token) and see how the shape changes and what they all mean. (In most paper or images, they often omit `nbatches` or `h` for clarity but I will explain including all the parameters) Also, to give a clear intuition of how everything is working, I will use the three following sentences I used above:

```Plain Text
I love you            --3
I am in love with you --6
I love you, too       --5
```


### Understand The Exact Starting Point
shape: `(nbatches, n_seq)`

So let's say the batch of sentences are already transformed into sequence of token IDs using the Tokenizer. This will be our starting point.

For example, based on [GPT-4o & GPT-4o mini tokenizer](https://platform.openai.com/tokenizer) provided by OpenAI

{{< note >}}
Note: the only reason we use the GPT-4o tokenizer here is since it's the most convenient tokenizer available on the web. However all the rest of the concepts and parameters (e.g. size of $vocab$) will be based on the original paper
{{< /note >}}

```Plain Text
I love you            --3
I am in love with you --6
I love you, too       --5
```

![[shaped-trasnformer-example-1.png]]


We use padding token(`PAD`) to keep the length of all `n_seq` same (which is crucial for matrix calculation). If we set `n_seq` to 6, the padding will fill as below:

![[shaped-trasnformer-example-2.png]]


These tokens converted to token ids will be

![[shaped-trasnformer-example-3.png]]

{{< note >}}
$Q$. Why is the shape `(nbatches, n_seq)` sometimes described as `(nbatches, n_seq, vocab)` if each token ID is just a scalar value?

$A$. In the original paper, the authors simply state that they use _learned embeddings_ to map token IDs to vectors of dimension $d_{model}$, without mentioning one-hot explicitly. 

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

So in real implementations, we directly use the `(nbatches, n_seq)` token ID tensor to _index_ into `W_emb` and fetch the corresponding rows.

**In practice:** think of `vocab` as the _number of unique token IDs (vocabulary size)_, not as an actual one-hot dimension in the input.
{{< /note >}}


### Token Embedding

**input shape: `(nbatches, n_seq)`**

Now we embed all token in sequences within a batch.

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
```

We add the PE with E for every token:
```mathematica
E+PE[0, 0, :] = [ 0.12+0.00, -0.07+1.00,  0.91+0.00, -0.04+1.00, ... ]   # "I"
           = [ 0.12, 0.93, 0.91, 0.96, 0.26, 1.03, -0.18, 1.55, ... ]

E+PE[0, 1, :] = [ 0.33+0.84, 0.25+0.54, -0.48+0.84, 0.18+0.54, ... ]     # "love"
           = [ 1.17, 0.79, 0.36, 0.72, 0.75, 0.95, 0.91, 0.32, ... ]

E+PE[0, 2, :] = [-0.55+0.91, 0.19-0.42, 0.07+0.91, 0.92-0.42, ... ]     # "you"
           = [ 0.36, -0.23, 0.98, 0.50, 0.77, -0.34, 1.27, -0.44, ... ]
```

As always **remember** this process is done by all sequences in the batch! (which means this process is parallelly done in the sentences "I am in love with you" and "I love you, too" as well!)


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

Let's break down how the tensor shapes transform in this process step-by-step.

{{< note >}}
From now, we will call the input as `x`.
{{< /note >}}


1. **Project to $Q, K, V$** `(nbatches, n_seq, d_model)` -> `(nbatches, n_seq, d_model)`
	We first need to get the $Q, K, V$ matrices. This is done by multiplying our input `x` with three learned weight matrices: `W_q`, `W_k`, and `W_v`. Each of the weight matrices have size of shape `(d_model, d_model)`.
	Each $Q, K, V$ becomes shape `(nbatches, n_seq, d_model)`
	`Q = x @ W_q = (nbatches, n_seq, d_model)`
	`K = x @ W_k = (nbatches, n_seq, d_model)`
	`V = x @ W_v = (nbatches, n_seq, d_model)`
{{< note type="tip" >}}
**A Note on Parameters:** We aren't training `Q`, `K`, and `V` directly. 
The actual parameters we train are the weights: `W_Q`, `W_K`, and `W_V`.
{{< /note >}}

2. **Splitting into Heads** `(nbatches, n_seq, d_model)` -> `(nbatches, n_seq, h, d_k)`
	After projection, we split `d_model` into `h` seperate heads. Since in the paper, d_k is defined as `d_k = d_model / h`, we can divide the last dimensions into `h` and `d_k` and view the shape as `(nbatches, n_seq, h, d_k)`.

3. **Transpose for Attention Calculation**
	To perform the attention calculation ($QK^T$) across all heads at once


Though this post is not aiming for high level explanation, to briefly explain, Multi-Head Attention means using multiple Attention heads when calculating Attention Weight. Each heads have their own learned weights($W_Q, W_K, W_V$) to calculate $Q,K,V$ matrices. For example, first head's $Q,K,V$ parameters can be represented as $W_{Q1}, K_{Q1}, V_{Q1}$ and $i$'th head's as $W_{Qi}, W_{Ki}, W_{Vi}$. In the paper, there are 8 heads in attention calculation: `h = 8`.

Now, let's look at how operation works in each head. Multi-head just means the operation in a single head done parallely on `h` heads independently. As mentioned above, each head in Multi-Head has three parameter set which are `W_q`, `W_k`, and `W_v`.
The inputs will be copied and matrix multiplied with `W_q`, `W_k`, and `W_v` each which respectively produces $Q, K$, and $V$ matrices.
(Note: we are not trying to train $Q, K, V$ itself, but $W_Q, W_K$ and $W_V$ are our actual parameters we're targetting).

Mathematically,
$$
\begin{aligned}
Q &= \text{x} W_Q \\
K &= \text{x} W_K \\
V &= \text{x} W_V
\end{aligned}
$$

Each shape of Ws in each head are `(d_model, d_k)`. So the total $W_Q, W_K$ and $W_V$ each can be  represented as `(d_model, d_k)`.

$Q$. Why is the shape not `(nbatches, n_seq, d_model, d_k)` but `(d_model, d_k)` instead?

$A$. This is because weights are parameters independent of the number of tokens(`nbatches` or `n_seq`). Batch size and sequence length are just broadcasted flexibly and dynamically depending on the input.

So we matrix multiply `(nbatches, n_seq, d_model)` with `(d_model, h*d_k)` and the shape of all $Q, K,$ and $V$ becomes `(nbatches, n_seq, d_k)`.

We would want to transpose and reshape this into `(nbatches, h, n_seq, d_k)`. This makes parallelization of calculating $QK^T$ for multiple heads easier.

$Q$ shape: `(nbatches, h, n_seq, d_k)`
$K^T$ shape: `(nbatches, h, d_k, n_seq)`

$QK^T$ is broadcasted into `(nbatches, h, n_seq, n_seq)` and this is the shape of attention map. So we have `h * nbatches` of attention map for each `n_seq`.

Then we calculate the attention weight with $W$:
$attn\_map = QK^T$ shape: `(nbatches, h, n_seq, n_seq)`
$W$ shape: `(nbatches, h, n_seq, d_k)`

This becomes `(nbatches, h, n_seq, d_k)` for each head.

After this process, we concatenate all the heads within a batch as `(nbatches, n_seq, h*d_k)`
Since `h*d_k == d_model`, we can express the output after the self attention layer as `(nbatches, n_seq, d_model)`


**Note on design choices:** 
- In practice, each attention head receives the **same full input** of shape `(nbatches, n_seq, d_model)`. The difference between heads comes entirely from their learned projection weights $(W_Q, W_K, W_V)$. Heads do not split the input; instead, they learn to focus on different subspaces of the same representation.
- Does `d_k` must need to be `d_model / h`? Also does `d_k` should always match `d_v`? No for both. It's not impossible. However setting `d_k != d_model / h` or `d_k != d_v` will lead to dimension mismatch or parameter disequilibrium, so it's just a practical and rational design choice. We can, however use different `d_k` and `d_v` by matching the output's projection or using feed forward network.


output shape: `(nbatches, n_seq, d_model)`



## 3. How Shape changes in code

Last part, I will show how the shape changes is implied in actual code (in Annotated Transformer, specifically). Since we have already went through the shape change intuition in section 2, it will be much easier to follow :)


## Conclusion

We have went through the common notations used in transformer, and used specific example to go through "one training phase" of Transformer Architecture. Finally we saw how shape changes in actual code (why all those `reshape` and `transpose` are used). Hope this post helps you gain insight and intuition of the bigger picture.
