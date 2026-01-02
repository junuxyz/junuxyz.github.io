+++
title = "(KR)Essence of Gradient Descent: Why and How"
date = 2025-09-03T21:48:22+09:00
draft = true
categories = ['ML']
+++

_Revisiting my older post (originally written in Febuary 10th, 2025)_
_Added some Python code for example._

---

In order to understand modern deep learning, we need to know exactly how gradient descent is applied.

If I explain gradient descent in one sentence, it's a method that iteratively moves toward the direction where $\cos \theta$ of each variables to the Loss Function.

There were total four points I was curious about when trying to understand gradient descent.

$Q_1$. Why do we use the expression "the steepest" about the direction? What does it mean?
$Q_2$. Why is it the most steepest direction when $\nabla f(a) \cdot u$ is the biggest?
$Q_3$. Why does gradient vector always direct to the steepest direction?
$Q_4$. Why does the Loss become minimal when we move towrad the direction where $\cos \theta$ becomes minimum.

### $Q_1$: 가장 가파르게 증가한다는 말이 왜 사용되는거야?

우선 '가장 가파르게 증가'한다는 말부터 짚어보자.

내가 이해가 가지 않았던 부분: 

이차함수를 예로 들면 $y = ax^2 +b$를 미분한 도함수는 $y' = 2ax$ 이고 특정 지점(x=b)에서 2ab라는 기울기를 **고정적**으로 갖는다. 

이처럼 내가 이해하고 있는 미분값은 '**고정이 된 하나의 값**'이다. 그래서 함수값이 '가장 가파르게 증가'한다는 말이 직관적으로 이해가 가지 않는다.

### $A_1:$

**1. 단일변수 함수에서도 '가장 가파르게 증가한다'는 말이 사용 가능한 이유**

단일변수 함수(Ex. $f=ax^2$)는 도함수가 $f' = 2ax$로, 한 점에서 움직일 수 있는 방향이 둘이다. (왼쪽 or 오른쪽) 

Ex. $f'(a) = 3$ 이라고 할 때 오른쪽으로 가면 3, 왼쪽으로 가면 -3이 된다.
단일선형회귀함수에서 경사하강법을 사용할 때도 오른쪽(3) 대신 더 작아지는 왼쪽(-3)을 선택하는 것이다.

**2. 다변수 함수의 경우**

변수가 하나만 더 늘어나도 (Ex. $f= ax + by$) 2차원 이상의 벡터가 된다. 

이때는 한 점에서 무한히 많은 방향으로 이동이 가능하다.

경사하강법은 (여러 방향으로 이동할 수 있지만 그중) **가장 가파르게 감소**하는 방향으로 이동하는 것에 관심을 가진다.

### $Q_2$ 왜 $\nabla f(a) \cdot u$가 가장 클 때가 곧 함수값이 가장 가파르게 증가하는 방향이야?

### $A_2$:

우선 $D_u$가 가장 클 때가 곧 함수값이 가장 가파르게 증가하는 방향이다.
왜냐하면 $D_u$ 자체가 함수의 변화량을 의미하기 때문이다.


테일러 급수를 이용해 다음과 같이 근사할 수 있다:

$$f(a+hu) \approx f(a) + h (\nabla f(a) \cdot u)$$
$$D_u f(a) = \lim_{h \to 0} \frac{f(a+hu) - f(a)}{h} = \lim_{h \to 0} \frac{f(a) + h (\nabla f(a) \cdot u) - f(a)}{h} = \lim_{h \to 0} \frac{h (\nabla f(a) \cdot u)}{h} = \nabla f(a) \cdot u$$


**따라서 $D_u f(a) = \nabla f(a) \cdot u$이고 $\nabla f(a) \cdot u$가 가장 클 때가 곧 함수값이 가장 가파르게 증가하는 방향인 것이다.**

### $Q_3$: 왜 Gradient Vector는 항상 함수값이 가장 가파르게 증가하는 방향을 가리켜? & $Q_4$: 왜 $cos \theta$가 가장 낮아지는 방향으로 갈 때 loss가 최소화 돼?

### $A_3$

이 두 질문에 대해서는는 **내적**(dot product)을 통해 직관적으로 이해가 가능하다.

우선 Gradient Vector ∇𝑓의 정의를 살펴보면 각 변수에 대한 함수의 편미분으로 구성된 벡터이다.

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right)$$

2변수 이상의 다변수 함수는 이론상 무한히 많은 방향으로 이동이 가능하기 때문에 가장 가파르게 감소하는 지점을 찾아야 한다. 이를 위해 단위벡터 $u$ 만큼의 방향으로 이동하는 것을 생각해보자.

(cf. 단위벡터란 벡터의 길이가 $1$로 고정된 벡터이다. 단위벡터 $u$를 사용하는 이유는 벡터의 크기를 고정해야 경사하강의 '방향'을 확인하기 더 수월하기 때문이다. 실제 이동하는 벡터는 단위 벡터 $u$에 학습률 $\alpha$를 곱한 값이다.)

벡터 a와 벡터 b의 내적은 $a \cdot b = |a||b| \cos \theta$ 이므로 단위벡터 $u$와 f의 내적은 $\nabla f \cdot u = |\nabla f||u| \cos \theta$ 이다.

이때 $|\nabla f|$는 Gradient Vector의 크기( $∇f$는 Gradient Vector이고 $|\nabla f|$는 그 크기임)이고 단위벡터 $u$는 1로 고정됐기 때문에 **유일한 변수는 $\cos \theta$가 된다.**

![[cosine_function.png]]
_image from Wolfram Function Site_

$\cos \theta$는 두 벡터가 이루는 각도를 말하는데 0도일때 최댓값인 1, 90도일 때 0, 180도(정반대 방향)일 때 -1이다.

학습률을 고려하지 않고 **방향**만 고려했을 때

$$\nabla f \cdot u = |\nabla f||u| \cos \theta$$
에서 
- $|\nabla f|$는 Gradient Vector의 크기
- $|u| = 1$
- $\cos \theta$가 가장 클 때의 $\cos \theta = 1$

따라서 가장 가파르게 증가하는 방향은 $|\nabla f| * 1 * 1$

즉, Gradient Vector의 방향이다.

<br>


### $A_4$:

경사가 가장 가파르게 내려가는 방향은 $\cos \theta = -1$일 때이고 이는 Gradient Vector와 정반대 방향인 것이다.

결국 $A_3$와 $A_4$의 핵심은 각 변수에 대한 함수의 편미분을 구했을때 $\cos \theta$ 를 제외하고는 모두 고정된 상수가 되고 $\theta$ 의 값에 따라 최대와 최소가 결정된다는 것이다.