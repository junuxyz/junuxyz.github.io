+++
title = "Gradient Descent and Backpropagation"
date = 2025-09-26T22:26:38+09:00
draft = true
categories = ['fundamentals']
+++

## What is Back Propagation?

This part explains the internals of what `backwards()` actually does in deep learning frameworks such as PyTorch.

Back Propagation, [introduced by Geoffrey Hinton and his colleagues](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) is a clever way to reuse the deriviation of local derivatives to calculate preliminary(early) layers.

In fact there are other ways to calculate the gradient for each and every weights. For example, technically we _can_ directly calculate the point where gradient converges to 0 by (visual) The problem is it takes A LOT of time. (O complexity)

Below is a simple Python script showing how inefficient this is and why Back Propagation is so good:


Another representative way is using **numerical gradient**.


### Numerical Gradient with NumPy

수식

numerical gradient simply means setting a very small numerical value, $\epsilon$ (e.g. $\epsilon = 0.00..01$) and calculate the gradient of each weights manually. Of course this is tedious and computationally inefficient work to do since we would need to calculate every weights independently.

### Chain Rule

Back Propagation solves this by reusing the gradients from the next layer. 
Back Propagation (especially This is available because of the chain rule.

The definition of chain rule:
chain rule 수식


### How PyTorch Back Propagation Works


### What is Optimizing?

This part explains the internals of what `optim()` actually does in deep learning frameworks such as PyTorch.

{{< note >}}
Optimize here, doesn't mean 최적화. It's a word used to updating parameters.
{{< /note >}}


{{< note >}}
Remember: back propagation is NOT the sole way to calculate gradients of each weights. It's just an efficient way. So modularize and think independently for Back Propagation and Optimize(Parameter Update) step.
{{< /note >}}

Now suppose we got all the gradients of the weights in a vector format (since gradient is after all, a vector), either by Back Propagation or numerical gradient. If we look at a typical neural network image with multiple layers, it seems that all the weights in different layers are sparse to each other. Therefore they update in different time space.

(image of nn)

but in fact, when we say, we train a **deep neural network model, basically we are training a super high-dimension vector that has ALL the weights from the very first to very last**. This vector is a gradient that has all the information of how each weights independently affects to create the final output and thus get the loss value. Since the definition of gradient itself is the most rapidly increasing direction of the super high-dimension space, we use gradient descent(negative) and multiply scalar value $\alpha$ or learning rate to each and every elements in the vector.

Optimization 수식
scalar mul in $\alpha \times vector$ (very first weight of very first layer to very last weight of very last layer)

So tldr; while we calculate each local gradients in back propagation one layer by one layer, the final global gradient we get is a huge vector. Since vector geometrically can be interpreted as a _direction_, we can think of the gradient as a single direction in very high dimension that points to the most rapidly increasing direction. We want to decrease the loss so we multiply negative to it and multiply it by learning rate to adjust the size of the step.

Now, I believe you've got a better understanding of back propagation and optimization (or parameter update) which is the fundamental backbone of modern deep learning models. But there might be people who are still curious about the question below... Below I will write about why we can make sure gradient is the most rapidly increasing direction and why gradient descent is the most rapidly decreasing direction. We will mathematically prove this so it will be a bit mathy but I will try to make it understandable.

### Optional: Why is Gradient Descent the most rapidly decreasing direction?

This section is for people who want to really understand Gradient Descent, mathematically and why it's the definition of "the direction(vector/matrix) the Loss function decreases most rapidly". You can skip this part if you're not interested in these details.


Since $D_u$ is the 변화량 of a function, the value of the function increases the most when $D_u$ is the biggest.

From Taylor 급수, $f(a+hu) \approx f(a) + h (\nabla f(a) \cdot u)$ 

$$D_u f(a) = \lim_{h \to 0} \frac{f(a+hu) - f(a)}{h} = \lim_{h \to 0} \frac{f(a) + h (\nabla f(a) \cdot u) - f(a)}{h} = \lim_{h \to 0} \frac{h (\nabla f(a) \cdot u)}{h} = \nabla f(a) \cdot u$$

Therefore, $D_u f(a) = \nabla f(a) \cdot u$이고 $\nabla f(a) \cdot u$ 가 가장 클 때가 곧 함수값이 가장 가파르게 증가하는 방향인 것이다.


If we look at the definition of Gradient vector $\nabla f$, it's 편미분 of each features.

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right)$$
(편미분 for $n$ features)

Since functions with features more than two(e.g. $L = ax + by + cz$) can theoretically move to infinite direction, we need to find the direction that points the lowest.

Since we only want to consider the direction, not the step size, let's use 단위벡터 $u$ as the step size.

{{< note >}}
단위벡터 means vector with the size fixed to 1.
{{< /note >}}

Since vector $a$ and vector $b$'s dot product is $a \cdot b = |a||b| \cos \theta$ , 
in the same manner dot product of $\nabla f$ and $u$ will be $\nabla f \cdot u = |\nabla f||u| \cos \theta$.

$|\nabla f|$ is the gradient vector itself
$u$ is vector of size 1
so the only variable that changes is $\cos \theta$ which depends on parameter $\theta$.


$\cos \theta$ means the 두 벡터가 이루는 각도, and as we know, $\cos$ is biggest when $\theta = 0$ which becomes 1 and smallest when $\theta=180$ which becomes -1.

Therefore the fastest increasing direction would be
$$|\nabla f| \times 1 \times 1 = \nabla f$$ Of course the fastest decreasing direction would be 

$$|\nabla f| \times 1 \times -1 = -\nabla f$$

## Methods on Optimization

### SGD

### SGD + Momentum

### RMSProp

### Adam


