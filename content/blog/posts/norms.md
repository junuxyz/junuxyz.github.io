+++
title = "Let's Talk About Normalization"
date = 2026-01-03T10:01:38+09:00
draft = true
categories = ['ML']
tags = ['fundamentals']
+++

## What is "Normalization" in the literature of Deep Learning?

In Deep Learning, _**Normalization**_ is a technique used to standardize the data being fed into and through a neural network.

The **main idea is to transform the features to be on a similar scale**. This is done by adjusting the range of the data, often to have a mean of zero and a standard deviation of one.

This process is important because it helps to **stabilize and speed up the training** of deep neural networks. By ensuring that the inputs to each layer have a consistent distribution, normalization can prevent issues like vanishing or exploding gradients, which can hinder the learning process. Ultimately, this leads to more reliable and faster convergence to a good solution.

## When do we Normalize

First of all, _What are we exactly normalizing?_

It depends on the type. But generally, we are trying to normalize the input of each layer before implementing activation function to it during the forward pass.

![[Pasted image 20260103100705.png]]

## Types of Normalization Layer

In this section, we will go through some of the most popular Normalizations and see in which task they are typically preferred and why.

![[Pasted image 20260103100628.png]]

## Batch Norm

## Layer Norm

Learn parameters that let us scale / shift the input data

1. **normalize the input data**
    
    $$ \begin{array}{c} \mu_{i} = \frac{1}{K} \sum_{k = 1}^{K} x_{i , k} \\\\ \sigma_{i}^{2} = \frac{1}{K} \sum_{k = 1}^{K} \left(\right. x_{i , k} - \mu_{i} \left.\right)^{2} \\\\\hat{x}_{i , k} = \frac{x_{i , k} - \mu_{i}}{\sqrt{\sigma_{i}^{2} + \epsilon}}\end{array} $$
    
2. **scale / shift them with learned param**
    
    $$ y_{i} = \gamma \hat{x}_{i} + \beta \equiv \text{LN}_{\gamma , \beta} \left(\right. x_{i} \left.\right) $$
    

This is done by each element of a Batch.

I've borrowed the mathematical expressions from [https://leimao.github.io/blog/Layer-Normalization/](https://leimao.github.io/blog/Layer-Normalization/) Thank you!

## A walk-through example

**Step 1. Normalize the Input Data**

For example, suppose we have 2 images per batch: $B = \{x_1, x_2\}$

and for each image, let’s say it’s a 2x2 sized image (to keep things simple) e.g. $x_1 = \{1,3,5,7\}$

For a flatten format of each image, we can calculate its mean $\mu$ and variance $\sigma^{2}$.

For $x_1$,

$$ \begin{array}{c} \mu_{1} = \frac{1}{K} \sum_{k = 1}^{K} x_{1 , k} = \frac{1+3+5+7}{4} = 4 \\\\ \sigma_{1}^{2} = \frac{1}{K} \sum_{k = 1}^{K} \left(\right. x_{1 , k} - \mu_{1} \left.\right)^{2} = (1-4)^2 + (3-4)^2 + (5-4)^2 + (7-4)^2 = 20 \end{array} $$

$x_2$ will get its own $\mu_2, \sigma_2$ depending on its element.

Based on the $\mu$ and $\sigma$, we normalize it ($\epsilon$ is added to avoid [**division by zero**](https://en.wikipedia.org/wiki/Division_by_zero) problem):

$$ \hat{x}_{1,k} = \frac{x_{1,k} - \mu_1}{\sqrt{\sigma_1^2 + \epsilon}} $$

For $x_1$, it would chage as $\hat{x}_i = \{-0.15, -0.05, 0.05, 0.15\}$

**Step 2. Scale / Shift it with learned parameters**

$$ y_{1} = \gamma \hat{x}_{1} + \beta \equiv \text{LN}_{\gamma , \beta} \left(\right. x_{1} \left.\right) $$

<aside> 💡

The important part to notice is that this is done independently by each inputs in a batch. So if we have $N$ inputs in a batch, we get $N$ pairs of $\gamma$ and $\beta$.

</aside>