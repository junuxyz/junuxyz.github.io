+++
title = "Fluent Python Cheat Sheet for Newbies"
date = 2025-07-22T18:50:06+09:00
draft = true
categories = ['ML']
tags = ['ml', 'python']
+++

High-level ML frameworks and libraries (e.g., PyTorch, JAX, TensorFlow, NumPy, Triton) are mostly based on Python.

I've known Python for a while, but I've never learned it to a professional degree and wouldn't say I'm good at Python programming. So, I decided to read _Fluent Python_ (which seems to be one of the 'bible' figures of Python) to cover some topics and improve my Python programming skills.

One thing that was hard for me while reading this book was that it was too dense to just understand and skip. I needed to type in the code, but even that wasn't enough. I figured this is a dictionary-like book that is hard to read repeatedly, so I decided to write a much more compact, digestable, and straightforward course note that works as a cheat sheet but also keeps the context as much as possible.

Let's start.

## Part 2. Data Structures

A _sequence_ in Python is a general term for an ordered collection of items. This means the items have a specific order, and you can access them by their position (their index).

We can divide sequences into types that can hold items of different type(_Container sequence_) or types that can't (_Flat sequence_). Flat sequences are more compact but are limited to holding primitive values.

Another way to divide is by mutability(_Mutable sequences_ vs _Immutable sequences_).


### Listcomps and Genexps

**List Comprehension, or Listcomp**
this creates list with single line without using `for` loop or `.append()`

```shell
>>> arr = [1,2,3]  
>>> ex = [-x for x in arr]  
>>> ex  
[-1, -2, -3]
```

listcomp is even faster than `map` or `filter` at least in some cases.
(Try running [this](https://github.com/fluentpython/example-code/blob/master/02-array-seq/listcomp_speed.py) in Python Shell if you are curious!)


**Cartesian Product**
this is just nested listcomp (double for loop).

```shell
>>> colors = ['black', 'white']
>>> sizes = ['S', 'M', 'L']
>>> tshirts = [(color, size) for color in colors for size in sizes]  
>>> tshirts
[('black', 'S'), ('black', 'M'), ('black', 'L'), ('white', 'S'),
('white', 'M'), ('white', 'L')]
```


**Generator Expression, or Genexps**

what this is/does: similar to listcomps but uses parantheses instead of square brackets. This **saves memory** because it **yields** items one by one using the iterator protocol instead of building a whole list and proceeding.

listcomp is expensive but reusable, generator is cheap but cannot be used again.


### Tuples

**Tuples as Records**
Since tuples are immutable, they (to be more accurate, their location) can be used as records.

**Tuple Unpacking**
we can unpack tuples that have two or more values by assigning items from an iterable to a tuple of variables.

This is an example of tuple used as a record and unpacking it.

```shell
>>> names = [('junu', 'park'), ('john', 'doe')]  
>>> for first_name, _ in names:  
... print(first_name)  
...  
junu  
john
```

we can use `*args` while unpacking tuple to extract the part(s) of what we want.

```shell
>>> names = [('junu', 'park'), ('john', 'doe'), ('ruby', 'onRails'), ('elon', 'musk'), ('the', 'primagen')]
>>> a, b, *c = names
>>> a
('junu', 'park')
>>> b
('john', 'doe')
>>> c
[('ruby', 'onRails'), ('elon', 'musk'), ('the', 'primagen')]
```

note that we can only use one args each time.


**Nested Tuple Unpacking**

If the expression of nested tuples match, Python will match the nesting structure and unpack it properly.

this is a non-realistic example but you will understand how nested tuple works by this example:

```shell
>>> nested_tuple = [('a', ('b', ('c', ('d')))), ('e', ('f', ('g', ('h'))))]
>>> for (x, (y, (z, (w)))) in nested_tuple:
...     print(x, y, z, w)
...
a b c d
e f g h
```


**Named Tuples**
This enhances the names(field names and a class name) of tuple.

```shell
>>> from collections import namedtuple
>>> Name = namedtuple('Name', 'first_name last_name')
>>> junu = Name('Junu', 'Park')
>>> elon = Name('Elon', 'Musk')
>>> prime = Name('The', 'Primagen')
>>> junu
Name(first_name='Junu', last_name='Park')
>>> elon
Name(first_name='Elon', last_name='Musk')
>>> prime
Name(first_name='The', last_name='Primagen')
```

As you can see there are two parameters for namedtuple: first one is the class name and second are field names seperated by single space.

