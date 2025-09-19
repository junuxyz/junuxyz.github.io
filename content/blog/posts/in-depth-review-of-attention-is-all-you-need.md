+++
title = "In-depth Review of Attention is All You Need"
date = 2025-08-16T15:09:09+09:00
draft = true
categories = ['ML']
tags = ['Paper Review']
+++

# 1. Abstract

seq2seq models before were all RNN or CNN based, which decodes the input in a sequential manner. This leads to inefficient, time-consuming process.

> what are seq2seq models?

Before we move on, to explain shortly about seq2seq models, or Sequence to Sequence models,

The paper introduces a new architecture called Transformers to only depend on the attention mechanism without recurrence (RNN) or convolution(CNN).

Transformers architecture
- became the new SOTA on seq2seq models
- used clever techniques and architectural decisions to enhance parallel processing which significantly reduced the model's training time.
- was more generalizable than other models.


# 2. How Transformer Work

We are going to look at how Transformer Architecture works based on the image shown above, starting from the inputs and explore all the components until the model's output(output probabilities).

I will try to explain in understandable words, and as detailed as possible.

## 2.1.1. Tokenization