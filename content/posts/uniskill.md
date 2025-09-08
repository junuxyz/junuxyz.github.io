+++
title = "[Paper Review] UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representation"
date = 2025-08-26T13:04:39+09:00
draft = false
categories = ['ML']
tags = ['Paper Review', 'Robot Learning', 'Video Learning']
+++

![[unkskill.png]]
# Abstract

Imitating experts is challenging due to visual, physical differences between human and robot.

Previous methods used cross-embodiment datasets with shared scenes and tasks but these data are limited which makes it hard to scale.

This paper presents a new framework called **UniSkill** that learns embodiment-agnostic skill representation from large video dataset.

# 1. Introduction

UniSkill uses image-editing pipeline for the neural network to focus on capturing the dynamics changes (over static content) between temproally distant video frames.

Since the dynamic changes in the vidoes are _motion,_ the model learns the motion(dynamic) patterns as skill representations.

This enables the use of embodiment-agnostic video dataset, which means it includes videos performed by humans, leading to a lot more available training data.

Experiment result:
- UniSkill effectively learned cross-embodiment skill representation
- generalized to unseen human prompts at test time without additional guide
- enhance robustness to new objects and tasks
- performance improved as more data sources were added
in BOTH simulation **and real world.**

# 2. Related Work

## Latent Action Models

Derived action-relevant information through inverse or forward dynamics models.

| Method       | Approach                                                                                                                         | Limitation/Differentiator                                                                            |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| LAPO, Genie  | Learn generative interactive environments with latent actions from gameplay videos.                                              | These methods are primarily tailored to game settings with discrete actions.                         |
| LAPA         | Extends latent action models to real-world robotic manipulation by incorporating diverse videos, including human demonstrations. | The learned latent actions are used indirectly, only to pretrain the policy as pseudo action labels. |
| **UniSkill** | Treats latent actions as explicit skill representations.                                                                         | Directly trains a skill-conditioned policy on the learned representations.                           |

## Explicit Action Representation

Transfers action information from human videos to robots via explicit action representations, such as 2D/3D trajectories and flow fields.

| Method                             | Approach                                                                             | Limitation/Differentiator                                                                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MimicPlay, EgoMimic, Motion Tracks | Extract 3D human hand trajectories from multi-view videos or wearable sensor inputs. | These methods often require calibrated cameras, pose tracking, or environment-specific constraints, limiting their scalability to off-the-shelf video datasets. |
| ATM, Im2Flow2Act                   | Predict 2D motion paths or flows from task-labeled human videos.                     | These methods often require calibrated cameras, pose tracking, or environment-specific constraints, limiting their scalability to off-the-shelf video datasets. |
| **UniSkill**                       | Avoids any task-specific trajectory extraction or pose supervision.                  | Learns directly from raw RGB videos, which enables the use of diverse public human and robot datasets.                                                          |

## Cross-Embodiment Skill Discovery

[XSkill](https://arxiv.org/abs/2307.09955) is the most similar approach as UniSkill, using cross-embodiment for skill discovery.
Here's how it differs:

| Method       | Approach                                                                                                | Limitation/Differentiator                                                                                                                                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| XSkill       | Aligns skills from human and robot videos via Sinkhorn-Knopp clustering.                                | This clustering with shared prototypes implicitly assumes some degree of alignment between human and robot videos. Human videos must cover the target robot task and be captured in similar environments for effective skill transfer. |
| **UniSkill** | Takes a different approach by **learning predictive representations through future frame forecasting.** | This completely removes the need for domain or task alignment, allowing the model to benefit even from entirely unrelated human videos.                                                                                                |

# 3. Method

## 3.1. Problem Formulation

### **1. Cross-embodiment video datasets** ($D_u = \{T_n\}^{N_a}_{n=1}$)

Videos that show both humans and robots doing things.

### **2. Robot demonstration datasets** ($D_a = \{T_n\}^{N_a}_{n=1}$)

Specifically robot videos that are "action-labeled," which means they have information about what actions the robot took at each step.

$D_a$ is relatively smaller than $D_u$ ($N_u >> N_a$).
This is because there are a lot more unlabled humand/robot video data than high quality, curated, and action-labled robot video data.

## 3.2: Universal Skill Representation Learning

### $ISD$ (Inverse Skill Dynamics Model)

$$ z_t = ISD(I_t, I_{t+k}) $$

- **Input:** Two temporally distant frames ($I_t$ and $I_{t+k}$) from a video $V$.
- Problem: relying soley on RGB pixels/frames lead to encoding of embodiment-specific(in this case, human) details, which can hinder the learning of embodiment-agnostic behavior $z_t$.
- to bypass this problem, UniSkill **internally** uses a depth estimation module to get depth information from these frames($I_t, I_{t+k}$).
- This helps the model to understand scene’s dynamics and spatial relationships. As an output, we get a universal skill representation, $z_t$.

### $FSD$ (Forward Skill Dynamics Model)

$$ I_{t+k} = FSD(I_t, z_t). $$

- predicts the future frame $I_{t+k}$ by given $I_t$ and $z_t$ from ISD.
    - So basically UniSkill is using a form of self-supervised learning to train a universal skill representation($z_t$) that effecitvely encodes the dynamic changes between two video frames.
    - There would be minimal (~close to no) changes in the difference of $I_t$ and $I_{t+k}$ which are only $k$ frames apart, besides the embodiment’s dynamic components.
    - To prevent $z_t$ from naively assign $z_t = I_{t+k}$, we enforce an information bottleneck on $z_t$.
    - Specific method: reforming as an image prediction task. Same architecture/method as InstructPix2Pix but instead of using language instruction, uses $z_t$ instead.

## 3.3. Universal Skill-Conditioned Policy

How the robot is trained to exectue the skills it learned in the previous step.

This uses the smaller, high-quality, action-labled robot dataset $D_a$

#### $$\phi^* = \text{argmax}_\phi E(o_t, o_{t+h}, a_{t:t+h}) \sim D_a [\log \pi_\phi(a_{t:t+h} | o_t, z_t)].$$

### Breaking down the formula

- $ϕ^∗$: This represents the **optimal set** of parameters for the policy network ($π$). The goal of the training process is to find these parameters.
- $\text{argmax}_ϕ$: This means we are trying to find the value of $ϕ$ that maximizes the expression that follows it. In simple terms, we are searching for the best possible policy parameters.
- $E(o_t,o_{t+h},a_{t:t+h}) \sim D_a$: This part specifies the **data source** for training. It means that the training is done by sampling a triplet of data points:
    - $o_t$: The robot's observation at the current time step ($t$).
    - $o_{t+h}$: The robot's observation at a future time step ($t+h$), where $h$ is the action horizon.
    - $a_{t:t+h}$: The sequence of ground-truth (labled) actions from time $t$ to $t+h$.
    - The notation $\sim D_a$ indicates that these **samples are drawn from the robot demonstration dataset**, which is where the action labels are available.
- $[\logπ_ϕ(a_{t:t+h}∣o_t,z_t)]$: This is the core of the behavioral cloning objective.
    - $π_ϕ$: This is the policy network with parameters $ϕ$. It is the function that we are training.
    - $a_{t:t+h}$: The sequence of actions that the policy is trying to predict.
    - $∣o_t,z_t$: These are the inputs to the policy network. The policy is "conditioned on" the current observation ($o_t$) and the universal skill representation ($z_t$). The skill representation $z_t$ is extracted from the robot demonstration data using the pre-trained ISD model.
    - $\log$: The logarithm is typically used in the training objective for stability and because maximizing the log-likelihood is equivalent to maximizing the probability.

so basically the source/distribution is on the left, the policy trying to optimized is on the right.

## 3.4. Cross-Embodiment Imitation with Universal Skill Representation
