+++
title = "Reframing TD Learning in the Perspective of Residual Connection"
date = 2025-10-02T10:35:50+09:00
draft = true
categories = ['']
tags = ['']
+++


### Bellman Backup

In the literature of Reinforcement Learning, Bellman Backup is mathematically expressed as 
$$y = r + \gamma \max_{a'} \hat V(s')$$
where we update $y$ based on the current reward and next state's estimated reward (via $\hat{V}$).
While it seems a bit counterintuitive that this works, the reason why it does is because we are using current reward $r$ as a ground truth, while trying to make $V$ as precise as possible.

This can also be reframed as a residual connection, or vice versa.

### Residual Connection

Residual connection was first introduced in [ResNet](https://arxiv.org/abs/1512.03385) and now is a widely used module/technique in the deep learning literature. The core concept of it is, when passing an intermediate representation of each layer, instead of trying to estimate the whole representation every time, pass the intermediate reperesentation and only learn the additional parts, which makes learning much more efficient.

Mathematically put, it can be 
$$y = x + f(x)$$
where $f$ can be a perceptron with non-linear activation function.


### Reframing Bellman Backup as Residual Connection

Back to our bellman backup equation, we are using $r$ as a ground truth but what we're trying to improve is the estimation of future value(expressed as $V(s')$)

