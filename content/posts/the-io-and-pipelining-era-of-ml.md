+++
title = "The I/O and Pipelining Era of ML"
date = 2025-08-01T16:00:15+09:00
draft = false
categories = ['Thoughts']
tags = ['short']
+++

Back in the days, constructing and finding novel neural networks(like CNN, RNN, GAN and many more) and scaling it to become "deeper" was the trend in Deep Learning research. 

After Transformers came out and as researchers noticed the power of Transformers, I feel the research trend shifted a lot into industrial and engineering problems. Yes, there were and are still some researches focusing on new architectures (like [Mamba](https://arxiv.org/abs/2312.00752), [Titans](https://arxiv.org/abs/2501.00663) etc) but in general I feel the trend has changed.

Especially after the "ChatGPT moment" researchers are working on how to efficiently optimize and deploy transformers to serve them in low cost, low latency, and high accuracy. I am not sure if this will be a short term trend after another novel state-of-the-art architecture comes out or another paradigm appears beyond LLMs as [Yann LeCun said](https://x.com/ylecun/status/1911604721267114206) (e.g. World Model, Robotics etc). 

I feel at least in the near term the trend of efficiency and engineering will prevail. 
Frameworks like [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm) and optimization techniques of Transformers such as [PagedAttention](https://arxiv.org/abs/2309.06180), [FlashAttention](https://github.com/Dao-AILab/flash-attention) and tools like [Triton](https://github.com/triton-lang/triton) and [CUDA](https://en.wikipedia.org/wiki/CUDA) programming are getting much traction than few years before. 

If I am thinking in the right direction, the mental model should be focused on **I/O and pipelining**. 

This means we need to understand 
- how each process of training and deploying are done
- why does each steps even exist (can we reduce the steps?)
- what inputs produces what outputs
- identify the underlying bottleneck (cost of time or latency etc) and optimize it.