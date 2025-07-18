+++
title = "From Transistors to Code -- Summary notes for the Digital Design and Computer Architecture Course"
date = 2025-07-17T13:13:16+09:00
draft = true
categories = ['ML']
+++

# From Transistors to Code -- Summary notes for the Digital Design and Computer Architecture Course

One of the most important concepts in Computer Science is **_abstraction_**, which simplifies lower layers, making them easy to use. This allows individuals working with higher layers (in most cases) to avoid concerns about 1. what things are and 2. how things work "under the hood." This approach offers advantages such as convenience, simplicity, and efficiency.

However, knowing and understanding *what actually happens* beneath provides significant insights for designing systems, making trade-off decisions, and comprehending the root cause of issues (aka. problem-solving skills). All of these are very important skills to possess in the field of Computer Science.

So let's dive in to all the abstraction layers in a bottom-up manner, starting from transistors.

_Note 1: The content is primarily based on the lecture [Digital Design and Computer Architecture](https://www.youtube.com/playlist?list=PL5Q2soXY2Zi9Eo29LMgKVcaydS7V1zZW3) by Professor Onur Mutlu and its supplementary textbook [Digital Design and Computer Architecture]([Digital Design and Computer Architecture: Harris, David, Harris, Sarah: 9780123944245: Amazon.com: Books](https://www.amazon.com/Digital-Design-Computer-Architecture-Harris/dp/0123944244)(Harris & Harris). I highly recommend taking this course if you are interested in this subject; it is very informative and relatively easy to understand; a great intro lecture._

_Note 2: This blog post can be considered a form of abstraction of the course, as I have ommited trivial or less important parts and tried to keep it simple, compact, and easy to understand. The main goal of this post is to introduce all the layers, from bottom to top in accessible language._

## 1. Transistors

Transistors are the fundamental building block of modern computer.
Transistors are cheap, easy to add on, and very efficient.

In the text book 
We will not cover beneath this layer, which introduces to a whole new physical world of creating a transistor.

## 2. Logic Gates

Logic gates are constructed by transistors.


## 3. Logic Circuits

_Logic Circuits_ are combined with logic gates to create logic that are more complicated.
We use _boolean algebra_(or _boolean equations_) to express logic circuits.

There are two types of Logic Circuits: Combinational Logic and Sequential Logic. The biggest difference is whether it has memory. Combinational logic has and sequential logic does not.

Let's first dive into Combinational Logic Circuits.

## 3-1. Combinational Logic Circuits

Some terms to note:
- _Complement_: variable with a bar over it (eg. $\overline{A}$) which means invert
- _Literal_: variable or its complement
- _Implicant_: product (which means `AND` in boolean algebra) of literals
- _Minterm_: **product** that includes all input variables
- _Maxterm_: **sum** (which means `OR` in boolean algebra) that includes all input variables

### POS and SOP

_Truth table_ is a table that lists all the unique set of inputs and outputs. While truth tables are easy to understand and expresses all the unique states of the Boolean Function, it is a relatively expensive and inefficient representation. So we usually use a standardized form for a boolean expression. There are two standardized forms(also called _canonical representations_) which are _Sum of Products(SOP) form_ and _Product of Sums (POS) form_.

Sum of Product means sum of all cases where F is 1(True).
Product of Sum is the _DeMorgan_ of SOP of $\overline{F}$. But this is a bit harder way to understand.
Think of it like this: It means "unless input is any of these that lead to F=0, it is true". POS is the sum of product of of all the cases where F is 0. This means if the input is one of the cases where $F=0$, the result of POS will become 0. If the input is not any of the cases where $F=0$, POS will become 1.


Now let's look at two simple examples to strengthen our understanding on both:

For example,

|  A  |  B  |  F  |
| :-: | :-: | :-: |
|  0  |  0  |  0  |
|  0  |  1  |  0  |
|  1  |  0  |  0  |
|  1  |  1  |  1  |
this is a simple AND logic gate in truth table format.

We can express this in SOP by $F(A, B) = AB$
We can express the same table in POS by $F(A,B) = {(\overline{A} \overline{B})(\overline{A}B) (A \overline{B}})$ 

for a more complex case,

|  A  |  B  |  C  |  F  |
| :-: | :-: | :-: | :-: |
|  0  |  0  |  0  |  0  |
|  0  |  0  |  1  |  1  |
|  0  |  1  |  0  |  1  |
|  0  |  1  |  1  |  0  |
|  1  |  0  |  0  |  1  |
|  1  |  0  |  1  |  0  |
|  1  |  1  |  0  |  0  |
|  1  |  1  |  1  |  1  |

this can be represented in SOP by $F(A,B,C) = \overline A\overline B C + \overline A B\overline C + A \overline{B} \overline{C} + ABC$
We can express the same table in POS by $F(A,B,C) = $F(A,B,C) = (\overline A\overline B\overline C)(\overline A BC) (A \overline{B} {C}) (AB\overline{C})$


Note that SOP or POS does NOT directly lead to minimal form of boolean function. Rather both lead to two-level logic, either Sum(+1) of Product(+1) or Product(+1) of Sum(+1).

SOP is a maxterm form since it's "**Sum** of ..." and POS is a minterm form because it's "**Product** of ...".

### Logic Simplification (or _Minimization_)

There are multiple ways to express the same logic, and these variants lead to different hardware features.

By using Boolean Algebra, we can simplify the SOP or POS logic (which are the universal form) in a methodical way.

We can think of it like this: Truth table -> SOP/POS form -> Boolean Simplification Rules

For example, $Y = (\overline{A}\overline{B}\overline{C}) + (A\overline{B}\overline{C}) + (A\overline{A}C)$ can be simplified as $Y = (\overline{B}\overline{C}) + (A\overline{B})$ 
You can practice simplifying Boolean Algebra problems on this [website](https://www.boolean-algebra.com/) if you are interested.


### Basic Combinational Blocks

Combinational Building Blocks are a higher level abstraction of combinational logic, which hides the unnecessary gate-level details(eg. `AND`, `OR`, `NOT` etc), serving as a building block for more complex systems.





When ready to publish:
1. Add appropriate category: `categories = ['Thoughts']` or `['ML']`
2. Change `draft = false` in frontmatter
3. Git plugin will auto-commit and deploy