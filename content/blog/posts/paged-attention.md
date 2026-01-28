+++
title = "Brief Explanation of PagedAttention"
date = 2026-01-03T12:57:32+09:00
draft = true
categories = ['ML']
tags = ['Transformer', 'inference', 'vLLM']
+++

This is the very first post for the upcoming vLLM v1 Deep Dive Series. In this post, we will only see what PagedAttention is, without explicit code. Initially I was going to write a thorough explanation of this paper, but decided to reduce it to only introduce the core principle/idea. This was due to the major shift from V0 to V1 architecture, which may confuse the reader when reading the paper and seeing the code implementation in vLLM. Thus, this will be a brief post to add some background knowledge before diving into vLLM v1 architecture.

## Intro

The _PagedAttention_ paper is relatively easy to read and understand — even readers without much background in LLM inference can follow along. However, it can be overwhelming on first read because they explain many techniques and optimizations in one paper.

So in this post, I'll first explain the  single most important thing PagedAttention achieves, terminology and components that make up vLLM, and finally walk through how it actually works during multiple batches of requests happen with examples and visualizations. (I will explain in terms of how V1 architecture works in the examples)

## PagedAttention is trying to solve ineffecient memory usage during decode

If I put it in one sentence: PagedAttention (and vLLM engine) is essentially trying to utilize memory better than existing systems for llm serving models.

If you remember the [last post](https://www.junupark.xyz/blog/posts/prefill-vs-decode/) I’ve written, LLM Inference workloads have two different parts, which are Prefill and Decode. While there are several techniques such as shared prefix that enhances Prefill’s throughput, the very core solution of vLLM and PagedAttention is how they tried to solve memory-bound decoding problems.

Why do I emphasize on decode stage?
This is because decode stage is bound to memory. This is a problem because during decode, it only computes single vector with matrix (vector-matrix multiplication) which can't fully use GPU's computation power. The bottleneck becomes the memory management and time loading KV Cache.

The techniques introduced in the paper all essentially(e.g. beam search is important in AI code assistant) trying to mitigate memory-bound issues cleverly using CPU scheduling and introducing new abstraction of KV cache memory.

In order to solve the underutilization of compute in decode, we can batch requests. But then the use of memory becomes more critical. This is because KV Cache size grows quickly with the number of requests. As stated in the paper, KV cache is calculated as follows:
$$\text{KV Cache} =2 \cdot (\text{key} \times \text{value vectors}) \times d_{\text{hidden}} \times L_{\text{layers}} \times 2 \, \text{bytes} \times N_{\text{max\_tokens}}$$
According to the paper, in 13B parameter OPT model, a single request can be as much as **1.6GB**. This means in a 40GB A100 GPU (even excluding the memory for intermediate activation) the maximum request it can handle is only (40-26)/1.6 = 8.75 requests.

This means 상대적 importance becomes more important due to the increase of GPU FLOPS, which means the computation ability increased while memory utilized inefficiently.

It is very easy to waste memory when processing requests. This is a unique characteristic of modern LLMs -- we don't know how long the prompt & the decoding step will be in prior. If we naively set and allocate the maximum token length (remember that decode process usually ends after `<eos>` is sampled or exceeds maximum token length), this will make huge internal fragmentation on prompts that are very short. For example think of a scenario when a user say "hi"(1 token) to a chatbot and it replies "Hello there, what can I help you?"(9 tokens). While the total token actually used is 1+9 = 10, if the maximum token length was set to 2048, 2038 token-worth memory is occupied without being used.

So to summarize:
1. LLM workloads are getting more and more important in modern AI.
2. We want to serve LLM as efficiently as possible.
3. One way to do this is batching requests during decoding.
4. This is a memory-bound problem since batching decode requests will need lots of KV cache stored.

This is why we need a smart scheduling system that wastes memory usage as minimal as possible while batching requests to its maximum. While this does not contribute computationally, it makes the overall performance better by enhancing the throughput.


## Terminologies

Before we explain how this works, we will first look at the terminologies necessary to know. There are many components in this engine.

### GPU Memory(VRAM) and KV cache

Before diving in, it helps to have a clear picture of what's actually sitting in GPU memory when serving an LLM.

![[Pasted image 20260103113432.png]]
_This is a revisualization of the original paper(when serving an LLM with
13B parameters on NVIDIA A100) and of course the specific ratio of each parts may differe depending on the model weight._

The model weights(colored as purple) and other static allocations(colored as dark blue) are fixed — they don't change between requests. What _does_ change is the KV cache(colored as light blue), which stores the key and value vectors from past tokens so we don't recompute them. This is the part we are trying to optimize.


### Block

Block is a fundamental building block of the vLLM engine. For people who know Operating Systems, it exactly corresponds to Pages but it's okay if you don't know what Pages are.

PagedAttention divides the memory available for KV cache into KV blocks. Each block contain key and value vectors of fixed number of tokens. Number of tokens depend on the block size while vLLM's default block setting is set to 16 (we will see why this is the default size in later section).
The important characteristics of a block is that for each requests, blocks need to be logically contiguous but doesn't need to be contiguous in physical memory(GPU DRAM). This makes the use of memory much more flexible and effecient.

This is similar to how Operating Systems uses Pages to run multiple programs efficiently. The paper explains:
>  Contiguous logical pages can correspond to non-contiguous physical memory pages, allowing user programs to access memory as though it were contiguous.

Below is a conceptual image of block of a single GPU(notice we only divided the light blue part(kv cache)?). In each GPU, there is a GPU worker and it has block engine. Block Engine allocates contiguous chunk of GPU DRAM and divides it into physical KV blocks. 
It's important to know that block is a row, not an element. There isn't a name for the element inside a block since block is the smallest 단위. For simplicity suppose the total KV cache available is 8 blocks with a size of 4 each. Obviously real world GPU memory will be a lot more bigger (e.g. 10GB+ / single GPU).

![[Pasted image 20260104134935.png]]

### Engine Core

Scheduler is the brain of the system, which is in charge of most of the decisions are made. It decides whether there's enough memory to start a new request, which requests move from waiting -> running, when to pause (preempt) a request if memory gets tight.

It doesn't do the actual memory allocation but just makes decisions and other components do the actual work.

### KV Cache Manager

KV Cache Manager, also called as KV Block Manager, maintains a block table, which has the mapping information of logical KV blocks to physical KV block numbers and number of filled slots of it.



### The Flow

vLLM uses a centralized scheduler in CPU which coordinates and decides if there is enough memory blocks for the request and also handles the order of request to process. First the Scheduler allocates the request into the waiting queue and check if KV blocks are available for the request. If it is, the request is moved to the running Queue.
The Scheduler makes decisions about memory allocation (e.g., allocate blocks, deallocate, swap). Then GPU Block Allocator will allocate the blocks from the free list and KV Cache Manager updates the block tables accordingly. These block tables are then sent to GPU workers so they know which physical blocks to read or write. GPU workers does not manage their own memory but instead, depend on KV cache manager.

As the request is being processed (either in Prefill or Decode) the projected KV values for each tokens are saved to the preallocated KV block by the KV Cache Manager. During Prefill, each workers process the tokens in a parallel manner and the KV values generated is saved to Physical blocks through something called _Cache Engine_. During Decode, if a block is unfilled, the rest of the block is reserved for future tokens. Note that either Logical or Physical KV blocks, the tokens are filled from left to right. 

If the Decode step takes long and exceeds a block, the Scheduler will request KV Cache Manager to allocate another block. GPU Block Allocator will assign the new block from the free list and KV Cache Manager will update the block table accordingly. Then the new block to process will be sent from the Scheduler to the Workers. Of course there is possibility of the decode to end early before the block ends which will lead to internal memory fragmentation but this is bound to the block size. As you can see, due to the block division and seperating logical/physical KV blocks allow KV cache memory to grow _dynamically_ instead of preallocating maximum length tokens. (e.g., maximum 15 for block size of 16 so it's not too bad -- actually way better than wasting 2000+ tokens with older inference engines). 

{{< note >}}
**Optional reading: CPU Block Allocatior and swapping**
CPU Block Allocator and swapping existed in vLLM v0. This worked as follows:
When GPU memory is full in use(free list in GPU Blcok Allocator is empty), vLLM uses a swap out method, which KV Cache Manager moves the KV cache of the not-used-immediately request(s). This is decided by the Scheduler. Then that block becomes a free list. After GPU memory gets more free blocks, it will swap in from the CPU to GPU.

However v1 has largely deprecated KV cache swapping because the overhead of moving KV caches between GPU and CPU was often slower than just re-computing the tokens, so v1 focuses on recomputation rather than swap-in/swap-out.
{{< /note >}}

## Example

In the original V0 Engine, Prefill and Decode were considered as different workloads, thus treated differently. However V1 has  Check the comment from the author in the V1 Implementation issues: https://github.com/vllm-project/vllm/pull/9289#discussion_r1807703823


Suppose there was a request as "The cat sat on the" in a KV block size of 4 tokens. This is converted into 5 tokens(`the, cat, sat, on, the`) and prefilled. Each token's KV will be saved sequentially in logical block starting from Block 0, left to right. Since the sequence exceeds a single block, we add another block(Block 1) for the request and add the last token's(`the`) KV inside the KV block. the rest part of Block 1 is not used yet reserved for future tokens.
At the same time, suppose GPU block allocator allocates Block 0 of a Logical KV block to Block 7 of a Physical KV block and Block 1 of Logical KV block to Block 3. 
Then block manager creates and maintains a block table that maps Logical block 0 with Physical KV block 7, completely filled (`# of filled = 4`) and Logical block 1 with Physical KV block 3, while the `# of filled` is 1.
As explained above, the it works even the physical KV block is not contiguous!

Now suppose we decoded the prompt and sampled a new token `mat`. At the next decode stage, `mat`'s KV cache will be saved to the reserved spot of Block 1, which is the second position. Since mapping is done block-wise and we don't need additional block to store `mat`, we don't need to do anything else.


