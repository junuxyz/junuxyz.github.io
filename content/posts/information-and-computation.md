+++
title = "Information and Computation"
date = 2025-08-15T19:44:57+09:00
draft = true
categories = ['Thoughts']
+++

## Poor Definition of AI

Artificial Intelligence, or AI is a very abstracted word for such vast field the technology covers.

Even if we look at Dictionaries or Deep learning courses, it's still hard to capture the fundamental and essence of what AI really is.


## My Definition of AI

I define the fundamental of AI as "computation of represented information".

Let's break down what this means.

It is very interesting how AI can predict protein synthesis, win against the Go champions in the world, how it can compare cat vs dog images, and how it answers the user's query by "predicting the next token".

How are all these possible?

### Computation

Recently listened to a podcast of ~ , the CEO of Groq and one of the creators of Google's TPU chip design.

At ~ he basically says AI is fundamentally computation. As an example, AlphaGo's 37th move was available because Google TPU's speed to search through the MCTS algorithm was fast enough (it's known as 1/10,000 probable move for humans).

When the researchers back tested the game with GPU, **it didn't output what TPU accelerated AlphaGo chose**.

So, the more efficient and effective the calculation is

Jensen Huang, CEO of NVIDIA calls GPUs time machine in that case because it allows thousands of millions worth computation work almost instant. And that's how AI actually works.


### Representation of Information

then what does representation and information mean?

The best way to think of AI is a giant black box function.

If we put some kind of input into it, it outputs in some other kind.

For example, 
- supervised learning: we input dog and cat images and let neural network predict the probability of the input (most likely).
- unsupervised learning: we input tons of data and the neural network predict the groups of it (clustering)
- self-supervised learning: we input words/tokens and neural network output the next token.
- reinforcement learning: we input the state and action space and neural network output the next move to make.
etc..

Even the Transformers architecture, which is the most popular neural network architecture since it's published until now, was first built for the purpose of translating words/sentences.

The important part is, it's always some kind of output for input(information).

However we cannot use raw information right into neural networks since it will not **interpret** what the information means.

One analogy to help understanding is, think of Neural networks as a foreigner who can only interpret bits(binary digits; 1s and 0s)

That means we need to have some kind of **translating** process in order to **represent** information into computer-friendly bits.

For NLP, we use tokenization.
For Image, we use image pixels.
For Audio, we use RPM/pitch.
For RL, there is a concept called **action tokens**, which is available actions in token format.
And for output, we use techniques like one-hot encoding to **represent** information from human-friendly way to machine-friendly way.

This gives us huge insight because Neural networks can interpret all kinds of data(multi modal and beyond) if we just find the right way to represent it.

Another Podcast I listened to, Jensen Huang said if text to text (language translation) works, why can't text to action token (Robotics/RL) work?

Because after all, to neural network it DOESN'T matter. It's all just bits.

All neural network does is take input in binary format, process it with its trained parameters (lots of mat mul) and output. It doesn't care once the input is appropriate.


So back to the question, what is AI?

AI is a technology that computes representation of information.

And because it doesn't care which kind of input it is, we just need to
1. abstractify the information well (with minimum loss and captured well)
2. compute efficiently

and the neural network will do the rest of the work.