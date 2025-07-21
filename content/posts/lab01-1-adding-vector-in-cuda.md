+++
title = "Lab01.1. Adding Vector in CUDA & Shallow Dive into CUDA Programming"
date = 2025-07-21T16:34:21+09:00
draft = true
categories = ['ML']
tags = ['CUDA', 'labs']
+++

_This post is just the notes taken while reading [CUDA tutorial document](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) while aiming myself to understand how [vector add](https://www.junupark.xyz/posts/lab01-adding-vector/) works in CUDA kernels, so I am going to keep things relatively simple (this is written just for the sake to not overwhelm myself too much). All the content here can be found in the original document with in-depth style._

Let's start!

## Introduction

### What is GPU?

We can use multiple pages to explain what GPU is, but practically speaking, it's a specialized processing unit for parallel computation.
If it is something not that complex but just has many things to compute (such as adding or multiplying) we can leverage from GPUs for the speed.

Below is an illustration provided from the official docs which is a (very simple) comparison on GPU (left) vs CPU (right) which will help us understand the difference intuitively.

![[cuda-gpu-architecture.png]]
### Building Blocks of GPU Architecture

It is good to have a solid understanding of the hierarchy of a typical GPU. Let's breifly look at the building blocks in a bottom-up manner:

1. **Thread**: This is the fundamental unit in CUDA/GPU programming. It executes one instance of your kernel function and has its own private registers.
2. **Warp(=32 Threads)**: The smallest unit of scheduling and execution on an SM(or Streaming Multiprocessor). Threads within a warp execute in lockstep. This is an important hardware detail, not explicitly defined by the programmer in kernel launch but crucial for performance. 
3. **Thread Block(up to 1024 threads, typically multiple warps):** A group of threads that are guaranteed to reside on the same SM. Threads within a block can synchronize via `__syncthreads()` and share data quickly via on-chip shared memory.
4. **Grid (multiple thread blocks):** This is the complete collection of all thread blocks for a single kernel launch. Blocks within a grid are independent and can only communciate via the slower global memory. The GPU scheduler assigns these blocks to available SMs.

### What is CUDA?

CUDA is a programming language developed by NVIDIA, which enables programmers to write GPU kernels more easily, supporting C, C++, and Fortran etc as a high-level programming language.

Another helpful image from the official documentation that captures the layer:

![[cuda-gpu-layers.png]]

> At its core are three key abstractions — a hierarchy of thread groups, shared memories, and barrier synchronization — that are simply exposed to the programmer as a minimal set of language extensions.

It is also worth noting that code written in CUDA is automatically **scalable** which can execute on any number of multiprocessors.

<br>

## Programming Model

### [Kernels](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernels)

CUDA both supports C and C++ and C++ functions are specifically called *kernels*. 

Kernel is defined using the `__global__` declaration specifier while the number of CUDA threads that execute that kernel for a given kernel call is specified using ugly three brackets(`<<<...>>>`). We will see these A LOT in CUDA written codes.

```cpp
__global__ void VecAdd(float* A, float* B, float* C)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int main()
{
	...
	VecAdd<<<1, N>>>(A, B, C);
	...
}
```

### [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy)

`threadIdx` is a built-in 3-component vector, which is helpful to make 1~3 dimension(based on your needs/data type) thread blocks.

The index of the thread and its thread ID is
- for one-dimensional block: it's the same
- for two-dimensional block of size (Dx, Dy): is (x + yDx)
- for three-dimensional block of size (Dx, Dy, Dz): is (x + yDx + zDxDy)
 don't get this part for now tbh




### Reference

https://docs.nvidia.com/cuda/cuda-c-programming-guide/