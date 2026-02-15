+++
title = "A Servey on LLM Serving Systems"
date = 2026-01-17T15:16:32+09:00
draft = true
categories = ['ML']
tags = ['inference']
+++

_This article is part of the LLM System Deep Dive Series. Rather than a detailed PagedAttention paper review (which may confuse readers given vLLM's v0→v1 architectural shift), this post introduces key ideas and a mental model for LLM serving._

## Introduction and First Principles of LLM Inference

**Large Language Model (LLM) inference** refers to using a trained model to generate outputs from new inputs (prompts) – for example, an autoregressive model producing text one token at a time. Unlike training, inference is entirely forward-pass computation, but it is _sequential_ for generative LLMs: each new token requires a full model evaluation conditioned on all prior tokens. This serial dependence makes inference inherently latency-bound, since generating $K$ tokens typically means $K$ forward passes. Even on powerful GPUs, much of the hardware sits idle due to this token-by-token dependency. The challenge is compounded by the sheer size of modern LLMs (billions of parameters) and their memory footprints. 
In short, **LLM serving** – providing fast, cost-effective inference for users at scale – is hard because it must juggle massive model compute, strict latency requirements, and limited hardware memory.


![[Pasted image 20260117225015.png]]
We will use [gpt-oss-120B](https://openai.com/index/introducing-gpt-oss/)(an openweight LLM by OpenAI released in 2025.08) as an example throughout this post!

To appreciate the constraints, consider gpt-oss-120B: just storing its weights in 16-bit precision(16BF) takes ~240 GB of memory, far beyond a single GPU’s capacity.

![[Pasted image 20260117223335.png]]
Figure 1. Example image of gpt-oss-120B model served on 4 NVIDIA Hopper H100 GPUs. We need at least 3 H100 GPUs to just load the model, but typically that's not enough so we need at least ~4 H100 GPUs to run big model like gpt-oss-120B. I will explain KV Cache in later section. 


Meanwhile, generating each output token involves tens of trillions of floating-point operations, and typical applications demand multiple tokens per second of generation speed for interactivity. [1] Thus, production LLM serving must optimize along two axes:
- **Throughput**: maximize tokens generated per second (and per dollar of hardware) by fully utilizing parallel compute.
- **Latency**: minimize the response delay for each request (e.g. time to first token and seconds per token).
Meeting these goals requires system-level considerations beyond basic implementations. We need to manage the _memory hierarchy_ (GPU high-bandwidth memory vs CPU RAM vs disk) to fit large models and their intermediate states, and to exploit parallelism wherever possible despite the sequential nature of generation.

In the rest of this blog, we’ll build up the concepts and techniques that have emerged to tackle these challenges in the past few years – from efficient batching and memory management to model parallelism and beyond – and see how state-of-the-art systems like [vLLM](https://github.com/vllm-project) or [SGLang](https://github.com/sgl-project/sglang) integrate them to serve LLMs efficiently in production.

## The Memory Hierarchy Problem in LLM Inference

One fundamental hurdle for LLM inference is the memory bottleneck. An LLM’s weights already strain GPU memory (as seen in Figure 1), but equally problematic is the memory needed for the activations and state during generation – most notably the _key-value cache_, or [KV cache](https://huggingface.co/blog/not-lain/kv-caching).
The KV cache stores the past sequence’s hidden keys and values at each self-attention layer so that when generating the next token, the model can attend to all prior tokens without recomputing their representations.
This cache grows with the sequence length and number of layers. For a decoder-only model with $N$ layers and hidden size $H$, each token adds roughly $2 \times N \times H$ elements (one key and one value per layer). In FP16 (2 bytes per element), that is $4 \times N \times H$ bytes per token. For example, gpt-oss-120B ($36$ layers, $H\approx2880$) [2] with a 2048-token prompt uses on the order of _~0.85 GB_ of GPU memory _per sequence_ just for KV cache. This means without even considering additional costs during inference(CUDA overhead, activations, etc.), only up to ~141 requests can be possibly processed. Clearly, with many concurrent requests or long contexts, the cache alone can overwhelm a GPU.

{{< note >}}
- This is actually a significant improvement compared to LLaMa-13B which uses 1.5-1.7GB of GPU memory per sequence just for KV Cache. [3]
- More detailed explanation on KV Cache can be found in https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
{{< /note >}}

Crucially this memory _must be fast_. Reading/writing the KV cache happens on every decode phase, making it bandwidth-sensitive. If the cache is stored in GPU VRAM, memory access is fast but capacity is limited. On the other hand, if it is offloaded to CPU memory or disk, capacity grows but access becomes a major latency drag. This is the memory hierarchy trade-off – GPU memory is precious and must be carefully managed. Naively allocating a contiguous GPU buffer for each request’s maximum possible context is extremely wasteful. For example the vLLM team measured that typical frameworks utilized only 20–40% of allocated KV memory, with 60–80% effectively wasted. The waste comes from mainly two forms of _fragmentation_ [4]:
- Internal fragmentation: Each request reserves a large contiguous slot for the worst-case sequence length (e.g. maximum tokens). If a request ends early (say it only used 600 out of a 2048 token buffer), the unused space in its slot stays unusable(e.g. 70% in the prior case). On average, models often use only ~20–30% of their reserved slot, leaving 70–80% empty.
- External fragmentation: Over time, many requests of varying lengths create “holes” in GPU memory – free segments between allocated slots. Eventually the GPU may have enough total free memory but not in any one contiguous block to satisfy a new large request. The result is out-of-memory errors despite non-full memory, due to inability to find a contiguous slab.

![[Pasted image 20260117225852.png]]
_Figure 2. Image from [4]._

Ironically techniques that improve throughput by increasing concurrency can _worsen_ memory pressure. If we allow dozens of requests to generate tokens in parallel (to keep the GPU busy), that means dozens of KV caches all kept live on GPU. This means fine-grained parallel scheduling solves the compute bottleneck but amplifies the memory bottleneck – more active sequences => more total KV memory reserved. The tension between maximizing GPU utilization and staying within memory limits is central to LLM serving system design.

Modern solutions approach this by treating GPU memory less like a static array and more like an OS-managed resource. Just as operating systems use paging, caching, and eviction to efficiently use RAM, LLM servers apply similar ideas to GPU memory for the KV cache and even model weights:
- They paginate or chunk large memory blocks to allow dynamic allocation and reuse (addressing fragmentation).
- They swap infrequently-used data to lower memory tiers (CPU or NVMe) when GPU memory is tight, trading some latency for capacity.
- They use quantization to reduce memory per parameter or per cache element, fitting more into the same GPU space.

We will discuss each of these strategies in depth. First, let’s look at how to better utilize the GPU compute through smarter batching and scheduling, while keeping the above memory caveats in mind.

## Continuous Batching and Iteration-Level Scheduling

**Batching multiple inference requests together** is a classic way to increase throughput. It feeds the GPU larger matrices to multiply which keeps more SMs busy and amortizes overheads. However static request-level batching (bundling $N$ requests and processing them to completion as one block) can hurt latency. If one request in the batch is long-running, all others are stuck waiting for it to finish before the next batch can start. For LLMs, different prompts produce wildly different output lengths; a request needing 1,000 decoding steps will hold up another that only needed 10 if they were statically batched together. This leads to poor latency for the shorter jobs and also idle periods where the GPU isn’t optimally used (once some requests finish early, their portion of compute goes idle). In short, while the overall throughput increases, latency for shorter requests increases.

![[Pasted image 20260117162410.png]]
_Figure 3. Delayed batch due to long request._
_In naive approach, the process latency and throughput is bound to the longest request, underutilizing GPU resource._


**Iteration-level scheduling** is the breakthrough to solve this. Pioneered by [FrendliAI](https://friendli.ai/)'s [Orca](https://friendli.ai/research/orca) system [5], it changes the granularity of work from “one batch = requests processed to completion” to “one batch = one _iteration_ (one token step) for some requests.”
In Orca’s approach, the server maintains a _request pool_ of all active requests. At each generation step, it selects up to $B$ requests from the pool (e.g. $B=4$ for batch size 4) and runs the model’s forward pass for one token on each of those requests concurrently. In the next iteration, finished requests (those that either exceeded the configured max token length or got an `<eos>`(end-of-sequence) token) are removed from the pool immediately, and new incoming requests can be added – meaning the composition of each batch can change at every step. This is often called continuous batching or in-flight batching, as the batch is continuously refreshed rather than fixed from start to end.

![[Pasted image 20260117232541.png]]
_Figure 4. Iteration-level Scheduler of B=4 requests per each step._
_This enables continuous batching – request that ends early doesn't have to wait until long request finishes. This reduces latency of shorter requests and thus increase throughput by keeping the GPU busy._

There are two benefits for this approach:
1. No request ever waits for another’s long tail – as soon as a request finishes (or a new one arrives), the scheduler makes use of the freed batch slot in the _next_ iteration without delay.
2. GPU compute is kept busy (=utilized) – late-arriving requests don’t idle in a queue but can start being processed interleaved with ongoing ones. Orca demonstrated massive throughput gains from this fine-grained scheduling (e.g. up to 36× throughput improvement vs. static batching on a 175B model at equal latency).

Iteration-level batching requires careful handling of mixed phases. Prefill processes many tokens at once while decode processes just one per request. Naively, these can't share a batch since tensor shapes don't match.

To understand why, consider what a forward pass looks like. For a batch of requests, you'd typically stack inputs into a tensor of shape `[batch_size, seq_len, hidden_dim]`. But if request A is prefilling 50 tokens and request B is decoding 1 token, what is `seq_len`? You'd either need to pad request B to 50 tokens (49 tokens padded; wasteful) or handle ragged tensors (tensors with different size of rows = complicated and poorly supported by most frameworks).

{{< note >}}
For people unfamiliar of the tensor shape used in Transformer, I have a [prior article](/posts/shaped-transformer/) that explains how tensor shape changes and why in the Transformer architecture. Also a seperate article that explains [prefill vs decode](/posts/prefill-vs-decode/).
{{< /note >}}

Orca's solution was to use selective batching - batches together operations that have the same shape requirements (like dense layers), but splits out others like attention that depend on sequence length.
Concretely, linear layers don’t care about sequence length dimension and can treat a batch of different-$L$ sequences as one long list of tokens (effectively summing the lengths). Attention layers, on the other hand, _do_ require aligning on the same $L$ (due to $QK^T$ computations), so Orca uses a custom fused kernel that runs multiple attention operations in parallel for each sequence with separate thread blocks. This avoids launching separate small kernels per sequence and keeps efficiency high.

The outcome is that an iteration-level scheduler can batch heterogeneous requests in a single GPU pass with minimal overhead. Systems like vLLM and Hugging Face Text Generation Inference (TGI) adopted similar continuous batching strategies, as it’s “almost required for modern LLM serving” to avoid stalls.

Let’s address another aspect of batching: overlapping the heavy prefill computations with the lighter decode steps via _chunking_.

## Overlapping Prefill and Decode with Chunked Prefill

Recall that LLM inference has two phases:
1. the compute-bound prefill (processing the entire prompt in parallel) and
2. the memory-bound decode (generating tokens one at a time using the KV cache).

A naive continuous scheduler might simply prioritize any arriving prefill (to minimize that user’s time-to-first-token, or TTFT). But if a very large prompt comes in, running its entire prefill in one go will pause all other decodes for a significant time – users will observe their streaming outputs “stall” while someone else’s prompt is ingested. To mitigate this, chunked prefill [6] was introduced.

![[Pasted image 20260118190509.png]]
_Figure 5. Image from [SARATHI paper](https://arxiv.org/abs/2308.16369)._
_Long prefill requests without chunked prefills create bubbles in GPU._

The idea of chunked prefill is straightforward: _break the prompt processing into smaller chunks_, and interleave them with decode steps from other requests. For example, instead of one 2048-token prefill blocking everything until completion, we might split it into, say, 20 chunks of 100 tokens and 1 chunk containing 48 tokens. Note that chunk size has nothing to do with block size, while it's obvious to make the chunk size in the orders of magnitude of block size. The scheduler can execute one chunk, then do a round of decode steps for others, then the next chunk, and so on.

By chunking, other requests don’t sit completely idle during a long prefill – they get to advance by one token for each chunk processed. Users experience this as slow-down rather than freeze: their text generation might momentarily slow while a big prompt is being ingested, but not halt entirely.

The trade-off is that the prefill itself takes a bit longer (with overhead of multiple smaller executions instead of one big matmul) and the first token for that request is slightly delayed. However, overall throughput improves because the GPU’s compute and memory are more continuously utilized by mixing workloads.

There’s a tunable _chunk size_: smaller chunks -> more overlap (responsiveness) but more overhead; larger chunks -> fewer interruptions (faster prefill) but longer potential pauses for others. In practice, chunk sizes of a few hundred to a few thousand tokens are used. For instance, early vLLM used $512$ by default, later increasing it as they refined the strategy. Systems like [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) even implement _dynamic chunk sizing_ [7], adjusting chunk length on the fly based on GPU utilization to balance latency vs. throughput.

![[Pasted image 20260117161436.png|500]]
_Figure 6. Image from [NVIDIA developer blog](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/)_
_TensorRT-LLM's dynamic chunk_

The result of chunked prefill is significant throughput gains under mixed workloads. One report noted up to +50% token throughput with chunked prefill in a vLLM deployment, without hurting latency budgets. NVIDIA similarly reports chunking yields both higher utilization and shorter tail latencies, especially for long-context requests.

{{< note >}}
$Q.$ What is the exact difference between Continuous Batching and Chunked Prefill? 
$A.$ Both improve throughput and GPU utilization, but they solve different problems:
**Continuous Batching** changes scheduling granularity. Instead of waiting for an entire batch to complete, the scheduler operates per-iteration (per-token), allowing new requests to join and finished requests to leave at any step.
**Chunked Prefill** addresses prefill/decode interference. It splits long prompts into smaller chunks so that ongoing decode requests don't stall while a large prompt is being processed. They are complementary — continuous batching decides *when* requests enter/exit the batch, chunked prefill decides *how much* of a prefill to process at once.
{{< /note >}}

We must remember, though, as we increase concurrent requests, the memory load grows too (each active request has a KV cache). This motivates memory-saving techniques that we’ll discuss next. We will turn our attention back to the KV cache – the primary consumer of memory during inference – and the techniques to manage it efficiently (paging and beyond). 

## PagedAttention (vLLM)

While vLLM is a subset of all the techniques introduced so far, the key insight behind vLLM is that the traditional way of storing the KV cache (one big contiguous array per sequence) is suboptimal. vLLM applies classic operating systems ideas: _paging, fragmentation avoidance, and copy-on-write sharing_. Their solution, **PagedAttention**, treats each sequence’s cache as a collection of fixed-size blocks (like memory pages) rather than one monolithic buffer. During attention, these blocks are indexed via a mapping table so that the model still sees a logical contiguous sequence, but physically the data can live anywhere in memory.

{{< note >}}
As the 1st author of the paper mentioned himself, "v" in vLLM stands for "virtual".
[Related Issue](https://github.com/vllm-project/vllm/issues/835#issuecomment-1692569559)
{{< /note >}}

Breaking the cache into, say, blocks of $K$ tokens (16 tokens per block as default [8]) yields near-zero fragmentation: only the last partially-filled block of a sequence is wasted space, which is <$K$ tokens worth. All other unused “slots” beyond the sequence’s current length can be eliminated, because we don’t pre-reserve the max length up front – we allocate new blocks on demand as the sequence grows. When a request finishes, its blocks are freed back to a pool and can be reused for new sequences immediately, avoiding external fragmentation holes.
The vLLM paper reports <4% memory waste with PagedAttention, versus 20–80% waste in prior systems. In effect, KV cache memory utilization jumps to ~96% optimal, which is far more (or longer) sequences can fit in the same GPU memory than before.
![[Pasted image 20260118192616.png]]
![[Pasted image 20260118192712.png]]
_Figure 7 & 8: Efficient Memory usage due to block._

Another powerful feature enabled by paging is sharing of cache content between sequences. If two queries have a common prefix (e.g. identical system prompt or an initial conversation history in a chat), we can map both of their logical sequences to the same physical blocks for that prefix, instead of storing two copies. This is analogous to memory deduplication and copy-on-write (COW) in OS: vLLM maintains reference counts on blocks and will only copy if a sequence tries to modify a block that is shared. A practical use-case is batched sampling or beam search: we often generate multiple continuations from the same prompt. Under PagedAttention, the prompt’s KV cache is computed once and all $n$ samples share those blocks initially, drastically reducing memory (and compute) overhead for multi-output generation. The vLLM team measured up to 55% memory reduction for scenarios like beam search, translating to about 2.2× throughput improvement in those cases.

It’s worth noting that PagedAttention required a custom CUDA implementation of the attention mechanism. The kernel needs to gather keys/values from non-contiguous memory locations (following the block index table) and still perform efficient matrix multiply operations. The results have proven that this is possible with low overhead, and the benefits far outweigh any slight extra cost in the attention kernel.

By solving the memory fragmentation issue, vLLM unlocked much higher batch sizes (throughput) _without_ running out of GPU memory. In fact, vLLM reported 2–4× higher throughput than earlier systems like FasterTransformer and Orca at equal latency, especially shining for long sequences and large models where memory was the bottleneck. [3]

### KV Cache Reuse and RadixAttention (SGLang)

While vLLM’s paging allows sharing identical prefixes, the [SGLang](https://github.com/sgl-project/sglang) project from [LMSYS ORG](https://lmsys.org/) observes that many applications repeatedly query the model with overlapping inputs [9] (for example, an agent that asks multiple variations of a question, or a chat session with a persistent history). They introduced RadixAttention to automatically reuse KV caches across _different_ generation calls whenever possible. 

![[Pasted image 20260118221456.png]]
Figure: Image from [9]

The approach is to maintain a radix tree (a compressed prefix tree) of all token sequences whose KV caches have been computed. Each node in the tree corresponds to a prefix of a sequence and holds the KV cache for that prefix (stored in a paged manner, one page per token, similar to vLLM’s blocks). When a new request comes in, the system finds the longest prefix that matches an existing path in the tree and can skip computing those tokens again – it directly replays the cached keys/values from the tree. The new tokens are then added as new branches in the radix tree.

In real world example, suppose we already served a request with prompt: “You are a helpful assistant. User: Hello! Assistant: Hi!”. Now a new request comes with the same initial system prompt “You are a helpful assistant” plus some new conversation. The tree allows recognizing  common prefix and reusing its cache, saving those computations.

RadixAttention uses an LRU policy to evict least-used branches when GPU memory is full. It effectively operates like a cache for KV caches across requests, leveraging temporal locality in what prompts are being served. The authors report that this was not handled by previous systems without manual concatenation of prompts, whereas their runtime does it automatically for arbitrary patterns of reuse. In combination with continuous batching and paged memory, RadixAttention can yield very large speedups in scenarios like agent loops or few-shot prompting where the same prefixes appear repeatedly. In their benchmarks, SGLang achieved up to 5× higher throughput on complex LLM workflows compared to vLLM (which doesn’t automatically reuse across requests). 

Importantly, RadixAttention is complementary to the techniques we’ve discussed: it builds on PagedAttention’s ability to store KV in independent per-token blocks (so that prefix caches can be pieced together and shared easily). It also works with continuous scheduling – the scheduler is made aware of cache hits so it can skip those tokens’ computation. This is a glimpse of how newer systems go beyond optimizing a single model call, and start optimizing across _multiple calls_ in structured applications (hence “Structured Generation Language”). For production use, such cross-request optimization might be more situational, but it hints at the future direction of LLM serving where the system can intelligently avoid redundant work at all levels.

We now tackle quantization, a critical technique to reduce memory and also accelerate inference.

## Quantization for Serving: Lighter Weights, Faster Inference

Quantization involves using lower numerical precision to represent the model’s parameters (and sometimes activations) so that memory and compute requirements are reduced. In LLM serving, post-training weight quantization has seen huge adoption because it can shrink model size by 2× or more with minimal accuracy loss. For example, the GPTQ method can compress 175B models from 16-bit to 3-bit or 4-bit weights with _negligible_ change in output quality. This cuts the memory footprint by >4×, enabling a model like GPT-3 (175B) to be loaded on a single high-end GPU (e.g. 80GB A100 can just fit a 3-bit quantized 175B). GPTQ also reported end-to-end speedups of 3.2× on A100 (and up to 4.5× on memory-bandwidth-limited GPUs like A6000) vs FP16, thanks to reduced memory traffic.

For serving, popular precisions include int8, int4, and even mixed schemes (e.g. 8-bit for most weights, 16-bit for critical layers). NVIDIA’s TensorRT-LLM and FasterTransformer have int8 support, often requiring calibration or per-channel scaling to maintain accuracy. New research quantization schemes like AWQ and AutoGPTQ automate finding the optimal scaling for each weight block to preserve model accuracy in 4-bit formats. There’s also interest in FP8 (e.g. Hopper GPUs support FP8 matrix math) – vLLM has added support for FP8 inference as well. FP8 can be seen as a form of quantization (1-bit sign, 5-bit exponent, 2-bit mantissa in one variant) that hardware can natively accelerate.

Quantizing activations and the KV cache is another frontier. If the KV cache could be stored in say 8-bit without harming generation quality, one could halve memory use for long contexts or high concurrency. Research indicates some tolerance here, but careful handling is needed to avoid degrading the model’s understanding of context. NVIDIA recently introduced an NVFP4 (4-bit) format for KV cache in inference, along with techniques to maintain accuracy for long context usage. This area is evolving, and likely future systems will dynamically choose precision for different parts of the model to optimize speed vs. fidelity.

From a performance perspective, lower precision not only allows fitting models on smaller (or fewer) GPUs, but it also often enables use of **tensor core** instructions that massively speed up math throughput (int8 tensor cores can achieve 4× the ops of FP16 on NVIDIA GPUs). The caveat is that quantization error can accumulate, so one must ensure the model’s outputs don’t drift. Techniques like per-layer calibration, bias correction, and mixed-precision dequantization for sensitive layers can help. The bottom line is that quantization is an indispensable tool for production LLM serving: it reduces cost and can even improve latency. Many production deployments today run 8-bit or 4-bit models to serve more users per GPU.

## Other Optimizations Worth Noting

While ther are other techniques and optimizations such as specupative decoding and various parallelism (tensor parallelism, model parallelism etc.) that are worth an in-depth post we will not get into it deeply in this post, since I thought these are more related to compute efficiency and orthogonal to KV cache management.

Until ~2024, this was the core implementations of LLM serving systems. However, recent development shifted into a smarter way of using KV Caches.

# KV-Cache as a First-Class Resource (2024–2026)

In the past few years, large language model serving systems have shifted their focus beyond just eliminating memory fragmentation in the key–value (KV) cache. Today’s cutting-edge inference engines treat the KV cache as a managed, reusable resource – something to be cached, quantized, compressed, and even transferred across nodes, not merely a monolithic tensor in memory. This extension of our survey (in the style of Lilian Weng’s technical blogs) covers major 2024–early 2026 developments: from automatic prefix reuse (vLLM’s “PagedAttention” and SGLang’s RadixAttention) to low-bit KV caches, distributed cache sharing, and cache-aware scheduling. We’ll also highlight how frameworks like NVIDIA’s TensorRT-LLM and Hugging Face TGI v3 now incorporate these capabilities as standard.

## From Fragmentation Fixes to Reusable KV Caches

Early optimizations like vLLM’s PagedAttention (2023) tackled a pressing issue: memory fragmentation in the KV cache. Naively allocating a contiguous KV buffer per sequence wastes 60–80% of GPU memory due to unpredictable sequence lengths. PagedAttention solved this by breaking each sequence’s cache into fixed-size blocks (“pages”) that can live _non-contiguously_ in memory. This OS-inspired design virtually eliminated fragmentation (<4% waste) and allowed flexible allocation on demand, so the system can pack many sequences together efficiently. The immediate payoff was huge throughput gains – vLLM reported up to 24× higher tokens/s than Hugging Face Transformers by avoiding memory stalls.

However, simply _storing_ KV cache more efficiently was only the first step. By 2024, the conversation shifted to reusing KV caches across requests. If two queries share the same prefix (e.g. identical system prompt or conversation history), why recompute those keys/values from scratch? The KV cache began to be viewed as “cached computation” that could be retained and served to future requests, much like an OS caches disk blocks. In other words, LLM serving moved from “just prevent memory waste” to “treat the KV cache as a first-class, reusable artifact.” As one 2025 runtime comparison put it, _“the winners are the runtimes that treat KV as a first-class data structure to be paged, quantized, reused, and offloaded, not just a big tensor slapped into GPU memory.”_ This reframing set the stage for new techniques in automatic prefix sharing, KV-cache quantization, persistent cache stores, and distributed-aware scheduling.

## Automatic Prefix Reuse: Hash Tables vs. Radix Trees

One of the flagship features to emerge was **automatic prefix caching** – the ability to reuse KV cache slices when a new request’s prompt overlaps with a previous prompt. Two different approaches became prominent:

- vLLM’s Hash-Based Prefix Cache: Building on PagedAttention, vLLM (v0.7+, 2024) introduced an automatic prefix caching mechanism using a global hash table of KV blocks. Each fixed-size block of tokens from a sequence is identified by a hash of _its content plus the preceding tokens_, and vLLM maps this “block signature” to a physical memory location. If another request later needs the same sequence of tokens, vLLM simply points it to the cached block instead of recomputing. Blocks are reference-counted and can be evicted with an LRU-like policy when GPU memory is full. Notably, this design achieves prefix reuse without maintaining an explicit tree of tokens – all blocks are independent and shareable, akin to caching pages in an OS. The result is that common prefixes (e.g. a standard system prompt or a frequently used few-shot example) get computed once and reused by all subsequent queries, significantly cutting redundant work. For example, if a 10k-token prompt is repeated, vLLM’s cache can skip ~3.7 seconds of prefill compute on the second run, bringing time-to-first-token (TTFT) from 4.3s down to 0.6s in one reported case.
    
- SGLang’s RadixAttention (LMSys, 2024): Around the same time, the LMSys team (authors of Vicuna/FastChat) proposed RadixAttention as part of SGLang, a structured generation runtime. RadixAttention takes a more explicit data-structure approach: it retains KV caches in a radix tree (prefix tree) keyed by token sequences. As different prompts arrive, the runtime performs prefix matching against the tree – shared prefixes map to the same tree path, and only the new suffix tokens need to be processed. The tree grows branches for diverging continuations and prunes least-recently-used leaves when memory is tight (LRU eviction of entire subtrees). Crucially, RadixAttention was designed to handle complex, multi-call workflows (agents, chain-of-thought, etc.) where many intermediate prompts overlap. By keeping the radix tree on CPU for efficient book-keeping and the actual KV tensors on GPU in a paged layout, SGLang achieved impressive reuse. Their tech report shows cache hit rates from 50% up to 99% on complex agent loops, yielding 6.4× throughput and 3.7× latency improvements over naive systems in prefix-heavy workloads. The tree structure naturally captures _token-level sharing_ (even within blocks), at the cost of some additional metadata management.
    

Both strategies have their merits. **Block-hashing (vLLM)** keeps the caching transparent and memory management simple – any matching chunk of tokens can be reused, and the implementation piggybacks on the existing page allocator. It works especially well for standard usage where prompts are reused wholesale or with minor additions. **Radix trees (SGLang)**, on the other hand, offer fine-grained sharing and a clear view of prefix relationships, which can benefit complex multi-turn or branching scenarios. Interestingly, vLLM’s eviction policy was designed to mirror RadixAttention’s behavior: it evicts unused _leaf blocks_ first (those at the longest prefixes), which is exactly what a tree would evict under LRU. In practice, both approaches avoid redundant computation in chatbots and batch inference. Figure 1 below illustrates this in a multi-turn conversation: the purple “cached prefix” (system prompt + history) grows with each turn, so only the latest user query needs expensive prefill on each request.

![https://llm-d.ai/blog/kvcache-wins-you-can-see](blob:https://chatgpt.com/f30dd265-eea5-40be-af0a-6e55d0ffd6b1)

_Figure 1: Prefix reuse in multi-turn chat. After the first turn, the shared prefix (system prompt + previous Q&A) is cached. Subsequent turns hit the cache for all earlier tokens, drastically reducing the computation needed for each new query. Only the new tokens (e.g. “Query2”, “Query3”) incur a cache miss and must be computed._

By early 2025, **prefix caching had become a standard feature** in many serving engines. For example, **NVIDIA’s TensorRT-LLM** added _KV cache reuse_ support (with a similar notion of prefix-based sharing) in its execution engine. Hugging Face’s **Text Generation Inference (TGI) v3.0** introduced a long-prompt pipeline with _automatic prefix cache_ as well, so that chat history tokens aren’t recomputed each time. The effect is especially dramatic for long conversation threads or few-shot prompts: TGI v3 benchmarks showed up to **13× faster** processing on very long prompts (with reuse enabled) compared to earlier versions. In summary, the industry moved towards **built-in prefix sharing** – turning what used to be redundant memory into a valuable cache that yields real latency and throughput gains.

## Quantized KV Caches: 8-bit and 4-bit for Higher Concurrency

As KV caches grew in importance, their memory footprint became the next optimization target. Recall that storing keys and values for all past tokens is memory-intensive – e.g. a single 2048-token context in a 13B model can consume several GB in FP16 form. In 2024–2025, developers began **quantizing the KV cache** to lower precision (int8, int4, etc.), trading a tiny accuracy loss for massive memory savings.

The idea is straightforward: use 8-bit or 4-bit representations for the stored **K** and **V** tensors instead of 16-bit floats. This **cuts memory (and bandwidth) usage by 2×–4×**, allowing far more tokens or concurrent sequences to fit in GPU memory. For example, LMDeploy (an inference toolkit from the InternLM team) reports that switching KV cache from FP16 to int8 doubles the number of cache blocks you can keep, and int4 quadruples it. In practice, they observed a **~30% throughput boost with int8 and ~40% with int4** on Llama2-7B, thanks to higher batch sizes before running out of memory. Equation (1) gives a rough comparison of memory per token:

KV Memory per token  ∝  precision (bits)×hidden_dim.\text{KV Memory per token} \;\propto\; \text{precision (bits)} \times \text{hidden\_dim}.KV Memory per token∝precision (bits)×hidden_dim.

This means **FP16 uses 2 bytes** per value, **FP8/INT8 use 1 byte**, and **INT4 just 0.5 bytes**. Intuitively, a 4-bit cache lets you pack **4× more tokens** into the same GPU memory than a 16-bit cache, dramatically improving capacity for long contexts or multi-user batching.

Early implementations kept the attention computations in higher precision to avoid quality loss: the cache would be stored in int4/8, but upon retrieval, values are dequantized to FP16/BF16 for the matrix multiplies. This way, quantization only affects memory, not the math in the Transformer – a strategy that yielded negligible accuracy drops on most benchmarks. For instance, vLLM added support for **FP8** KV cache (using the standard 8-bit E4M3 format) and noted it “minimally degrades inference accuracy”. NVIDIA went further with a custom **NVFP4** (4-bit floating-point) format in late 2025: NVFP4 compresses the KV cache to **25% of FP16 size** (half the size of FP8) and showed <1% accuracy loss on long-context tasks. This enables _doubling_ the context length or batch size on their Blackwell GPUs with virtually no model quality impact. In fact, because smaller caches ease memory bandwidth pressure, NVFP4 actually improved cache _hit rates_ and latency in some cases (by fitting more context in fast memory).

Overall, **quantized KV caching has become a key lever for scaling throughput**. By 2025, most inference frameworks support it: TensorRT-LLM allows KV in INT8 or FP8 mode (and is experimenting with 4-bit), vLLM and TGI integrate FP8 and other quantization libraries (like bitsandbytes, GPTQ) for weight _and_ cache compression, and LMDeploy offers easy toggles for int4/int8 KV on any NVIDIA GPU (Volta or newer). The empirical takeaway is that **memory is often the bottleneck** for LLM inference – so shrinking the KV cache with low-bit encodings directly translates to higher concurrency and longer context windows at a given hardware budget. It’s a rare “free lunch” where careful quantization (especially with calibration) yields big performance gains with minimal model degradation.

## KV Cache as a Transferable Artifact: Compression and Multi-Node Scheduling

Even with efficient allocation, reuse, and quantization, KV caches remain large pieces of data. This raises a question: what if we could **share cache content across machines or sessions**, not just within a single server’s memory? Recent systems have started to enable exactly that – treating the KV cache as a **persistent, portable artifact** that can be serialized, moved, and strategically placed to maximize performance in distributed setups.

One line of work focuses on **persisting and transferring KV caches** for _later reuse_. For example, the **LMCache** ecosystem (2024–2025) treats cached prompts as valuable assets that can be stored outside the GPU. One of its tools, **CacheGen (SIGCOMM 2024)**, introduces a custom tensor encoding to **compress KV caches 3× smaller than even quantized form**. This compressed cache can be written to disk or even cloud storage (e.g. S3) and later **streamed back into GPU memory faster than recomputation**. In other words, if you’ve processed a chunk of context once, CacheGen lets you “save the work” in a compact form and reload it on demand. Their measurements show _Time-To-First-Token_ improving by up to 3× versus computing fresh every time, when caches are served from a local SSD or RAM cache. In **Figure 2**, we see an example comparison: a vLLM server without CacheGen takes ~4.3 seconds to produce the first token for a long prompt, whereas vLLM+LMCache (with CacheGen) returns in ~0.74 seconds – over **3× faster** and even outpacing other cache systems like Fireworks or DeepInfra.

![https://blog.lmcache.ai/en/2025/07/31/cachegen-store-your-kv-cache-on-disk-or-s3-load-blazingly-fast/](blob:https://chatgpt.com/296de50f-5e85-4135-9b78-24b1f6779182)

_Figure 2: Persistent KV-cache unlocks fast startup. _Time to first token_ for a long prompt, comparing vLLM alone vs. various caching setups. Storing and loading caches via LMCache+CacheGen yields a **3.2× reduction** in latency (orange bar) vs. vLLM’s in-memory cache, by avoiding re-processing shared context. It outperforms other solutions that lack CacheGen’s compression/streaming optimizations._

To achieve this, CacheGen and similar solutions (e.g. Hugging Face’s _DiskKV_ prototype, IBM’s experiments) treat the KV data as binary blobs that can be **serialized with minimal overhead**. CacheGen uses an efficient codec that leverages the distribution of attention matrices to shrink size (their paper reports **3.5×–4.3× compression** of KV tensors). Importantly, these systems maintain format compatibility so that a cache computed on one server (or offline) can be loaded into another server’s memory and plugged into the model’s next-token computation. This opens the door to **pre-computing frequent prompts** (e.g. long retrieval-augmented context) and reusing them across sessions or even sharing across replicas. Indeed, LMCache demonstrated “_CacheBlend_” for Retrieval-Augmented Generation (RAG): they precompute the KV for each retrieved document chunk so that at query time, the LLM can skip directly to combining those cached chunk embeddings, rather than processing N documents token-by-token. This made RAG **4.5× faster** in their experiments without hurting answer quality. It’s a powerful concept – treat each piece of context as _modular, cached data_ that can be flexibly mixed into new prompts, as long as any necessary cross-attention adjustments are handled (CacheBlend did a small amount of re-computation to account for interactions between chunks).

Another critical development is how **distributed LLM serving** handles KV-cache locality. In a multi-node deployment, naive load balancing can accidentally undermine all these caching gains. For example, say User A’s first request hits Server 1 and populates a KV cache for the prompt; if their second request (with the same prefix) is routed to Server 2, that server can’t benefit from the cached prefix on Server 1. This was a noted pitfall: “vLLM’s prefix caching breaks in distributed deployments – standard load balancers scatter related requests across pods, destroying cache locality and forcing expensive re-computation”. The solution emerging in late 2024 is **prefix-aware scheduling/routing**: the system (or a smart scheduler like **llm-d** from IBM) tracks which prefixes or conversations reside on which node, and routes subsequent queries accordingly. Instead of random or round-robin assignment, requests with a shared prefix are pinned to the same executor to maximize cache hits. This has been shown to yield _order-of-magnitude improvements_ in distributed environments. The llm-d project reported **57× faster responses and 2× throughput** when using precise cache-aware scheduling versus a naive load balancer. NVIDIA’s inference stack also supports this pattern: TensorRT-LLM exposes a **KV-cache event API** so that a cluster manager can query cache contents or receive updates, enabling coordinated routing decisions. In effect, the runtime provides “hints” to an upstream scheduler about cache state, achieving an eventually consistent view of where each prefix’s KV lives. Armed with that info, a distributed serving system can greatly increase the overall cache hit rate (and thus lower cost – note that some APIs charge 10× less for cached tokens than uncached).

It’s worth noting that an alternative to smart scheduling is to physically **transfer KV caches between nodes**, but this is often less efficient. KV tensors for large models can be tens or hundreds of megabytes – sending those over the network for each request can become a bottleneck, especially compared to the GPU’s local memory bandwidth. One has to either compress aggressively (which is where CacheGen can help for “warm starting” cold nodes) or avoid transfers in the critical path altogether. Therefore, most 2025-era systems prefer to **keep cache reuse local** through intelligent routing, and leverage compression/offload mainly for longer-term storage or multi-session persistence. We also see hybrid strategies: caches can spill to CPU memory or NVMe on the same machine (much faster than across network) when GPU memory fills up, as done in some hierarchical cache designs. Techniques like **circular buffers** (used in TensorRT-LLM) allow old cache entries to be gracefully evicted or offloaded once they’re not needed, making room for new contexts while still avoiding fragmentation. All these methods reinforce the view of KV cache as **fluid data** – it can be compressed, moved, evicted, or fused as needed, rather than tied to a single context in one runtime instance.

## 2026: KV-Cache Optimizations Are the New Normal

As of early 2026, the advancements above have converged into the **standard feature set of LLM serving engines**. Just as FP16 weights and Transformer fusion became expected optimizations, we now expect top-tier runtimes to handle KV caches intelligently. For example, NVIDIA’s TensorRT-LLM (part of their NeMo framework) supports _paged KV caching, prefix reuse, KV offloading to CPU, quantized cache formats (int8/FP8)_, and even a _sliding circular cache_ for windowed contexts. It provides APIs for custom eviction policies (e.g. keeping system prompts in cache longer via high priority) and hooks for multi-node cache coordination. Similarly, Hugging Face’s open-source **TGI v3** has caught up to include **prefix caching and long-input chunking** out-of-the-box, yielding competitive performance on chat workloads. TGI v3 uses paged-attention kernels under the hood (inspired by vLLM) for efficient block-wise memory use, and integrates with quantization libraries so you can serve models with 8-bit weights _and_ 8-bit KV if desired. Even frameworks like LMDeploy and FasterTransformer now advertise _“blocked KV cache with management and reuse”_ as a key feature, alongside low-bit support. In short, **KV-cache management has become a first-class citizen** in LLM inference, acknowledged as both a major performance lever and a complex resource to optimize.

Looking back, this evolution underscores how critical the KV cache is to efficient LLM serving. Initially just an internal buffer to avoid recomputing attention, it is now treated almost like an extension of the model. We partition it, index it, hash it, quantize it, save it, stream it, and schedule around it. By doing so, modern systems maximize the reuse of past computation – which is essential for latency (skip what you already did) and for cost (cached tokens are cheaper than new ones). The trend also highlights an interesting co-design between _algorithms and systems_: ideas from CPU caching, databases, and distributed systems are being applied to neural network runtime, with significant gains. As we move forward, we can expect further innovations like **learned cache eviction policies**, **cross-user cache sharing** (with proper privacy guards), and tighter integration of generation frameworks with cache servers or schedulers. The bottom line is that serving large language models efficiently is no longer just about the model weights or GPU compute kernels – it’s equally about managing the _dynamic state_ (KV cache) that accumulates as the model runs. By managing that state wisely – reusing it, shrinking it, and placing it where it’s needed – we unlock a new level of performance and scalability in LLM applications
## Putting It All Together

We’ve surveyed a range of techniques addressing different aspects of the LLM serving problem. In practice, state-of-the-art serving systems combine many of these to achieve maximum performance. Let’s briefly situate some well-known implementations in the landscape:

- **Hugging Face Text Generation Inference (TGI):** An open-source server widely used to deploy models like Llama2. TGI focuses on efficient CUDA kernels (built atop FasterTransformer), supports continuous batching (iteration-level scheduling) and recently added chunked prefill for better concurrency. It also supports quantized models (via bitsandbytes int8/int4) and multi-GPU sharding. Prior to vLLM, TGI was a top performer, and even now TGI benefits from some features like multi-model hosting that vLLM doesn’t target. Throughputs are excellent, though vLLM has reported 2–3× higher throughput in long-sequence scenarios thanks to its memory optimizations.
    
- **vLLM:** Arguably the current state of the art for single-model inference, vLLM’s distinguishing features are PagedAttention (drastically improved memory efficiency) and support for continuous batching and chunked prefill out-of-the-box. It integrates many optimizations: CUDA graphs, FlashAttention, various quantization schemes (GPTQ, AWQ, etc.), and even speculative decoding. In deployments like LMSys’s Vicuna chatbot, vLLM enabled 5× higher traffic on the same hardware by allowing more concurrent generations without OOM and keeping utilization high. One limitation is that vLLM is primarily optimized for single-node (multi-GPU) serving; it doesn’t yet handle multi-host distributed inference for extremely large models (but those cases can often be served with model distillation or quantization to fit on one node).
    
- **NVIDIA TensorRT-LLM:** This is NVIDIA’s closed-source (at time of writing) high-performance engine building on their TensorRT framework, specialized for LLMs. It incorporates many of the ideas discussed, often under different names: in-flight batching (their term for continuous batching), **dynamic batch resizing**, and the **chunked prefill** feature with dynamic chunk sizing we described. It heavily utilizes low-level optimizations on NVIDIA GPUs, like fused kernels and int8 quantization calibrated for LLMs. TensorRT-LLM is known for excellent throughput on supported hardware, sometimes at the cost of flexibility (models might need conversion to a TensorRT engine which is a non-trivial process). It has shown up to 5× perf boosts in system benchmarks that involve long input prompts, by combining KV cache management and chunked prefill. For production users on NVIDIA stacks, it’s a compelling choice if you can handle the deployment constraints.
    
- **FasterTransformer:** An open library of optimized Transformer inference kernels by NVIDIA, often serving as the backend for others (TGI uses FT under the hood). It provides multi-GPU and multi-node support and was among the first to do efficient sampling, beam search, etc., with custom kernels. However, FT alone did not implement iteration-level scheduling or paged KV; it was more focused on single-batch performance. Systems like Orca and vLLM compare against FasterTransformer as a baseline and outperform it significantly by adding the scheduling and memory management layers. Essentially, FT solves the _compute_ problem well but not the _multi-request scheduling_ problem – so modern systems wrap FT kernels with smarter schedulers.
    
- **DeepSpeed Inference:** From Microsoft, DeepSpeed not only offers training optimizations but also an inference engine with features like ZeRO-inference (partitioning weights across GPUs and streaming them in from CPU NVMe if needed) and custom kernels for transformer blocks. It supports **concurrent request streaming**, where multiple sequences are interleaved on one GPU, similar to continuous batching. One unique ability is offloading model weights to CPU memory with background pre-fetching such that only needed layers are in GPU just-in-time – this allows serving a model larger than GPU memory with some latency penalty. DeepSpeed was influential in Orca’s development (the Orca paper cites DeepSpeed’s pipeline engine) and is still used in some large-model deployments where multi-node or CPU-offload is needed.
    
- **SGLang (RadixAttention runtime):** The newcomer from LMSys, SGLang is less about raw single-model throughput and more about _coordination of multiple LLM calls_ (like an agent loop or complex prompt program). On workloads with many repeated subqueries or chain-of-thought branches, SGLang’s ability to cache and reuse KV across those calls can give it a big advantage. It also claims strong base throughput (2–5× vLLM in some tests) with a highly optimized backend and a new scheduling approach. SGLang is built on top of vLLM’s ideas (it uses PagedAttention, etc.) and extends them. It shows the trend of integrating the _application logic_ with the inference engine to eliminate inefficiencies across call boundaries.
    

In summary, the landscape of LLM serving systems is rich, but they all share the fundamental techniques we discussed, in different combinations. A taxonomy of optimizations would be:

- **Efficient Scheduling**: (static batching → continuous batching → chunked scheduling) – improves compute utilization and latency fairness.
    
- **Memory Management**: (contiguous alloc → paging → sharing/reuse) – allows high concurrency and long contexts without running out of memory.
    
- **Compute Parallelism**: (single GPU → tensor/pipeline multi-GPU → multi-node) – scales model size and throughput as needed.
    
- **Precision Optimizations**: (FP16 → int8/FP8 → int4) – trades a bit of accuracy for huge gains in speed and memory.
    
- **Algorithmic Speedups**: (standard decoding → speculative decoding, etc.) – reduces the total amount of work needed to generate outputs.
    
- **Low-level Kernel Optimizations**: (unfused ops → fused ops, FlashAttention, etc.) – squeezes maximum performance from the hardware per operation.
    

Critically, these techniques **interact**. The best systems carefully combine them: e.g. continuous batching increases memory pressure, so it must be paired with a solution like PagedAttention to avoid OOM. Chunked prefill increases overlap, which in turn keeps GPUs busier (throughput up) but needs adaptive chunk sizing to not hurt latency. Quantization can sometimes reduce the _relative_ benefit of speculative decoding (if the model is twice as fast after int4 quantization, a 2× speculative speedup might be less vital) – so an engineer might prioritize one over the other depending on where the bottleneck lies. In fact, understanding the **bottleneck** (compute-bound vs memory-bound vs latency-bound) in a given deployment is key to applying the right optimizations. Many of the advancements, like Orca and vLLM, came from identifying a shift: as models got bigger and batch scheduling got finer, **memory became the bottleneck** instead of compute, hence the focus on new memory management.

## Conclusion

LLM serving has rapidly evolved from straightforward batching of requests to a highly optimized pipeline of clever techniques. We started with the basics – why autoregressive inference is slow and memory-hungry – and explored the full spectrum of solutions now available. Modern systems leverage every level of improvement: they batch across requests dynamically to keep GPUs busy, overlap prompt processing with generation, cache and reuse computations aggressively, distribute models across hardware, use fewer bits for faster math, and even predict tokens ahead of time to shortcut the sequential process. The end result is that what once might have been 1 token per second on a single model can now be tens or hundreds of tokens per second served to many users concurrently on the same hardware, all while meeting interactive latency targets.

It’s instructive to reflect on the journey: Early transformer deployments (circa 2020) often used brute-force methods – e.g. serving GPT-3 with dozens of GPUs per query to get latency down, or limiting sequence lengths to fit memory. The **Orca paper (OSDI 2022)** introduced a major leap with iteration-level scheduling, eliminating batch idle time. **FasterTransformer and FasterSeq** provided the toolkit of fast kernels. **vLLM (SOSP 2023)** solved the looming memory crisis with PagedAttention, just in time as user prompts and context lengths grew. Now, approaches like **SGLang (2024)** and others are tackling the next frontier: optimizing across multiple model interactions and making the serving engine more application-aware.

For engineers and researchers, the takeaway is to view LLM serving as a **systems problem** as much as an ML problem. We bring in concepts from OS (paging, scheduling), databases (caching), distributed systems (sharding, consistency of results), and hardware (quantization, specialized libraries) to make AI models usable in real-time products. Each piece of the stack contributes to the end performance. The extensive bibliography of techniques we’ve discussed – from **Orca** to **PagedAttention** to **speculative decoding** – provides a toolbox for anyone building or tuning an LLM service.

Looking ahead, we can expect even more integration: compilers that automatically apply these optimizations, hardware that provides finer-grained memory management for AI, and algorithms that adapt on the fly to workload patterns. The goal is always the same: maximize the useful work (delivering quality tokens to users) and minimize the wasted cycles or bytes. By standing on the shoulders of works like Lilian Weng’s deep dives and the research we cited, one can appreciate the elegant interplay of principles that allow gigantic models to talk to us in real time. The next time you interact with a chatbot or AI assistant, under the hood an entire serving system is orchestrating these optimizations to give you an answer within seconds – a little miracle of computer systems and machine learning co-design.


## References

[1] Prefill and Decode for Concurrent Requests - Optimizing LLM Performance https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests#:~:text=bot,prompt%20length%2C%20and%20concurrent%20load
[2] gpt-oss model card https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf
[3] vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention https://blog.vllm.ai/2023/06/20/vllm.html#:~:text=referred%20to%20as%20KV%20cache,The%20KV%20cache%20is
[4] Kwon et al., _Efficient Memory Management for LLM Serving with PagedAttention_, SOSP 2023.
[5] Yu et al., _Orca: A Distributed Serving System for Transformer-Based Generative Models_, OSDI 2022.
[6] SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills https://arxiv.org/abs/2308.16369
[7] Streamlining AI Inference Performance and Deployment with NVIDIA TensorRT-LLM Chunked Prefillhttps //developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/
[7] Inside vLLM’s New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html
[8] vLLM Blog – _Easy, Fast, and Cheap LLM Serving with PagedAttention_, 2023. (Explains PagedAttention and shows 24× throughput vs HF Transformers)
- Hugging Face Blog – _Prefill and Decode for Concurrent Requests_, 2025. (Excellent discussion on static vs continuous batching, chunked prefill, and latency trade-offs)

[9] Fast and Expressive LLM Inference with RadixAttention and SGLang https://lmsys.org/blog/2024-01-17-sglang/
Zheng et al., _Fast and Expressive LLM Inference with RadixAttention and SGLang_, arXiv 2024. (Introduces SGLang and RadixAttention for KV cache reuse in complex workflows)
- Leviathan et al., _Fast Inference via Speculative Decoding_, ICML 2023. (Original speculative decoding paper, 2–3× speedups with a draft model)
- NVIDIA Technical Blog – _TensorRT-LLM Chunked Prefill_, 2024. (Details chunked prefill implementation and benefits in NVIDIA’s inference stack)
- NVIDIA Technical Blog – _Introduction to Speculative Decoding_, 2025. (Overview of speculative decoding and the EAGLE method on NVIDIA GPUs)
- Frantar et al., _GPTQ: Accurate Post-Training Quantization for GPT_, 2022. (Advanced 3-4 bit quantization achieving <1% perplexity degradation on GPT-175B and >3× speedups)
- Data Science Dojo Blog – _Memory is the Real Bottleneck: How PagedAttention Powers vLLM_, 2023. (Great intuition on KV cache fragmentation and why fine-grained batching increased memory pressure)
- vLLM GitHub README (vLLM features and supported techniques)
- Hugging Face TGI Documentation – (Continuous batching and recent features like KV offloading, not explicitly cited above but background knowledge)