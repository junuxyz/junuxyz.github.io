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

_Note: This is not designed to be a replacement of the book, but for a handy review note or cheat sheet used daily._

Let's start.

# Part 2. Data Structures

## Sequences

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

We can use additional methods supported in namedtuples such as,

```shell
Name(first_name='The', last_name='Primagen')
>>> prime.last_name
'Primagen'
>>> prime._asdict()
{'first_name': 'The', 'last_name': 'Primagen'}
```


### Slicing

You can slice not only lists but other sequence types like strings or tuples.

**Why slices and range exclude last item**
because it works well with the zero-based(starting from 0) indexing used in Python.

More specifically, it's
- easy to see the length of a slice or range when only the stop position is given.
- easy to compute the length of a slice or range when start and stop are given by subtracting them.
- easy to split a sequence in two parts at any index `x` using the same `x` without overlapping.


**Using $+$ and $*$ with Sequences**

Multiplying list with integer copies of the same sequence works.

```shell
>>> l = [1,2,3]  
>>> l2 = l * 3  
>>> l2  
[1, 2, 3, 1, 2, 3, 1, 2, 3]  
>>> id(l)  
139738068947456  
>>> id(l2)  
139738067406976
```

**Building Lists of Lists**

The problem appears when the elements in the list are mutable items. The most common case is building lists of lists.

```shell
>>> board1 = [['_'] * 3 for i in range(3)]  
>>> board1  
[['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]  
>>> board1[1][2] = 'x'  
>>> board1  
[['_', '_', '_'], ['_', '_', 'x'], ['_', '_', '_']]
>>> id(board1[0])
139738042877824
>>> id(board1[1])
139738043505344
>>> id(board1[2])
139738043329344
```

board1 works fine, meaning it has different address for each sub arrays.

but if we multiply list, it just references the same list:

```shell
>>> board2 = [['_'] * 3] * 3
>>> board2
[['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
>>> board[1][2] = 'x'
>>> board2
[['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
>>> board2[1][2] = 'x'
>>> board2
[['_', '_', 'x'], ['_', '_', 'x'], ['_', '_', 'x']]
>>> id(board2[0])
139738065760064
>>> id(board2[1])
139738065760064
>>> id(board2[2])
139738065760064
```

which means it just appends the same row three times.


**Augmented Assignments** (`+=` and `*=`)

augmented assignments are NOT the same as adding or multiplying.
While `+` internaly uses special function `__add__()`, `+=` uses special function `__iadd__()`.
**Only if `__iadd__()` is not implemented in the class, it works the same as `__add__()`.**

so, when does it change?
In mutable sequences (such as lists), `__iadd__()` will be changed **in place** (using less memory) but in other cases, it will work as `__add__()` which means `a = a + b`.

```shell
>>> l = [1,2,3]
>>> id(l)
139738065799168
>>> mul_list = l * 3 # __mul__()
>>> id(mul_list)
139738068947456 # different address
>>> l *= 3 # __imul__()
>>> id(l)
139738065799168 # changed in place
```

**A+= Assignment Puzzler**


```shell
>>> t = (1, 2, [30, 40])
>>> t[2] += [50, 60]
```

This is a corner case that I won't go into details since it seems too peripheral.
To explain breifly, this will both make an error but also augment the list into `[30, 40, 50, 60]`.

> • Putting mutable items in tuples is not a good idea.
> • **Augmented assignment is not an atomic operation**—we just saw it throwing an exception after doing part of its job.
> • Inspecting Python bytecode is not too difficult, and is often helpful to see what is going on under the hood.


### Sorting

There are mainly two built-in sorting functions, which is `list.sort` and `sorted`.

`list.sort` does create a copy, but sort the list in place. It returns `None` to remind it.

Note: This is a nice pattern to know about Python API conventions: Python usually outputs `None` to make it clear the function doesn't create a copy but changes the object in place.

`sorted` creates a new list and returns it. Therefore, takes additional memory. This also means it can take any kinds of iterable object including immutable ones, such as tuples. Note that it always outputs a new **list**.

Both of the functions take two arguments which are `reversed: bool` and `key: string`.

```shell
>>> l = ['a', 'b', 'c']
>>> sorted_l = sorted(l) # example of `sorted`
>>> id(l)
139738067138496
>>> sorted_l
['a', 'b', 'c']
>>> id(sorted_l)
139738065799168 # another array created
>>> l.sort() # example of `list.sort`
```

Sorting is very helpful because once the sequence is sorted, each elements can be very efficiently searched.

### Managing Ordered Sequence with `bisect`

It is good to keep the sorted sequence since sorting is **expensive**.
`bisect.bisect` and `bisect.insort` helps a much efficient way to search and insert than `list` functions.

[Binary Search Algorithm](https://en.wikipedia.org/wiki/Binary_search) is provided via `bisect` module in Python standard library.

Bisect is an efficient way to search and insert elements into a sequence **while keeping the order** in an ordered sequence.

`bisect()` returns the location where the element should be inserted in order to maintain the order while `insort()` finds the location and inserts the element.

a quick minimal example would be:

```shell
>>> import bisect
>>> l = [1, 2, 3, 4, 5]
>>> l
[1, 2, 3, 4, 5]
# bisect only tells you the location
>>> bisect.bisect(l, 6)
5
>>> bisect.bisect(l, 3)
3
>>> bisect.bisect(l, 4)
4
# insort actually inserts the array
# this is an in-place exchange so only one block of
# memory will be added
>>> bisect.insort(l, 3)
>>> l
[1, 2, 3, 3, 4, 5]
>>> bisect.bisect(l, 3)
4
```

### When a List is Not the Answer

**Arrays**

When creating an array, 
1. you provide a typecode
2. a letter to determine the underlying C type used to store each item in the array.

For example

```shell
>>> from array import array
>>> from random import random
>>> floats = array ('d', (random() for i in range(2**10)))
```
'b': signed char
'd': double

whatever you assign to, the array will interpret as the type you assigned to it.

Fast loading and saving are available via `.frombytes` and `.tofile`
 Binary file can be 60x faster than reading numbers in text file.


**Memory Views**

> A memoryview is essentially a generalized NumPy array structure in Python itself (without the math). It allows you to share memory between data-structures (things like PIL images, SQLlite databases, NumPy arrays, etc.) without first copying.

**NumPy and SciPy**

**Deque**

`deque` (not _dequeue_) stands for 'double-ended queue'.

Removing items from the middle of a `deque` is not as fast. It's optimized for appending and popping from the end. Used in LIFO queue.

deque internally is implemented as a doubly linked list of fixed-size blocks. Therefore traversing to the middle would be more ineffecient than accesing the ends.

<br>
## Dicts and Sets

> Hash tables are the engines behind Python's high-performance dicts.

All mapping types in the std library use the basic dict in their implementation, which means all keys(while values are not required) must be _hashable_.

- atomic immutable types (such as `str`)
- `frozen set`
- `tuple` is hashable only if all its items are hashable


```shell
>>> tl = (1,2,[30,40])
>>> hash(tl)
Traceback (most recent call last): 
File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list'
>>> tf = (1,2,frozenset([30,40])) 
>>> hash(tf)
5149391500123939311
```



duck typing means Python doesn't care what something is only cares what it can do.

```shell
>>> a = dict(one=1, two=2, three=3)
>>> b = {'one': 1, 'two': 2, 'three': 3}
# combine multiple iterable objects element-wise
>>> c = dict(zip(['one', 'two', 'three'], [1,2,3]))
>>> d = dict([('two',2), ('one', 1), ('three', 3)])
>>> e = dict({'three': 3, 'one': 1, 'two': 2})
>>> a == b == c == d == e  
True
```

