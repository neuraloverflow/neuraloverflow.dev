---
layout: post
title:  "Python🐍 3.7 beginner's guide 👌"
---

### NOTE: This is still a guide in draft ... apologies 🙇

## Intro

We cannot talk about Python🐍 before discussing its philosophy: the Zen🏯 of Python🐍. To read it we can just type:


{% highlight python%}
import this
{% endhighlight %}

    >>> The Zen of Python, by Tim Peters
    
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!


Python🐍 is a dynamically typed language, which means that variable names can point to objects of any type. For example:


{% highlight python %}
x = [] # x is firstly defined as an empty list
x = 1  # now x is an integer
x = 'this is a string' # now x is a string
{% endhighlight %}

This does not conclude that Python🐍 doesn't have types. Python🐍 HAS types❗ However, the types are linked not to the variable names but *to the objects themselves*. Yes everything in Python🐍 is an object.

|Type|Example| Description|
|--------|------|--------|
| ``int`` | ``x = 1``  | integers (i.e., whole numbers) |
| ``float`` | ``x = 1.0`` | floating-point numbers (i.e., real numbers) |
| ``complex`` | ``x = 1 + 2j`` | Complex numbers (i.e., numbers with real and imaginary part)|
| ``bool`` | ``x = True``   | Boolean: True/False values|
| ``str`` | ``x = 'abc'``  | String: characters or text|
| ``NoneType``| ``x = None``   | Special object indicating nulls|

We'll take a quick look at each of these in turn.


{% highlight python %}
x = []
type(x)
{% endhighlight %}

    >>> list




{% highlight python %}
x = 1
type(x)
{% endhighlight %}

    >>> int




{% highlight python %}
x = 'this is a string'
type(x)
{% endhighlight %}




    >>> str



In object-oriented programming languages like Python🐍, an *object* is an entity that contains data along with associated metadata and/or functionality.
In Python🐍 everything is an object, which means every entity has some metadata (called *attributes*) and associated functionality (called *methods*).
These attributes and methods are accessed via the dot syntax.


{% highlight python %}
x = []
x.append(4)
print(x)
{% endhighlight %}

    >>> [4]


## Working with Strings

You can use quotes and apostrophes inside strings!


{% highlight python %}
"This is a string"
{% endhighlight %}

     >>> 'This is a string'




{% highlight python %}
"This is a string with 'quotes'!"
{% endhighlight %}

    >>> "This is a string with 'quotes'!"




{% highlight python %}
str = "This is a string"
print(str.title())
{% endhighlight %}

    >>> This Is A String



{% highlight python %}
print(str.upper())
print(str.lower())
{% endhighlight %}

    >>> THIS IS A STRING
    >>> this is a string


## Arithmetic Operations 🤖
Python🐍 implements seven basic binary arithmetic operators, two of which can double as unary operators.
They are summarized in the following table

| Operator     | Name           | Description                                            |
|--------------|----------------|--------------------------------------------------------|
| ``a + b``    | Addition       | Sum of ``a`` and ``b``                                 |
| ``a - b``    | Subtraction    | Difference of ``a`` and ``b``                          |
| ``a * b``    | Multiplication | Product of ``a`` and ``b``                             |
| ``a / b``    | True division  | Quotient of ``a`` and ``b``                            |
| ``a // b``   | Floor division | Quotient of ``a`` and ``b``, removing fractional parts |
| ``a % b``    | Modulus        | Integer remainder after division of ``a`` by ``b``     |
| ``a ** b``   | Exponentiation |  ``a`` raised to the power of ``b``                     |
| ``-a``       | Negation       | The negative of ``a``                                  |
| ``+a``       | Unary plus     | ``a`` unchanged (rarely used)                          |


## Bitwise Operations 🤖
In addition to the standard numerical operations, Python🐍 includes operators to perform bitwise logical operations on integers.
These are much less commonly used than the standard arithmetic operations, but it's useful to know that they exist.
The six bitwise operators are summarized in the following table:

| Operator     | Name            | Description                                 |
|--------------|-----------------|---------------------------------------------|
| ``a & b``    | Bitwise AND,     | Bits defined in both ``a`` and ``b``        |
| <code>a &#124; b</code>| Bitwise OR,      | Bits defined in ``a`` or ``b`` or both      |
| ``a ^ b``    | Bitwise XOR,     | Bits defined in ``a`` or ``b`` but not both |
| ``a << b``   | Bit shift left,  | Shift bits of ``a`` left by ``b`` units     |
| ``a >> b``   | Bit shift right, | Shift bits of ``a`` right by ``b`` units    |
| ``~a``       | Bitwise NOT,     | Bitwise negation of ``a``                          |

These bitwise operators only make sense in terms of the binary representation of numbers, which you can see using the built-in ``bin`` function:

## Assignment operators 🤖

To assign a value in Python🐍 we don't have only the operator = . Below 👇🏼 a table with all the Python🐍  assignment operators:

|||||
|-|-|
|``a += b``,| ``a -= b``,|``a *= b``,| ``a /= b``|
|``a //= b``,| ``a %= b``,|``a **= b``,|``a &= b``|
|<code>a &#124;= b</code>,| ``a ^= b``,|``a <<= b``,| ``a >>= b``|

## Comparison Operations 🤖

Another type of operation which can be very useful is comparison of different values.
For this, Python🐍 implements standard comparison operators, which return Boolean values ``True`` and ``False``.
The comparison operations are listed in the following table:

| Operation     | Description                       || Operation     | Description                          |
|---------------|-----------------------------------||---------------|--------------------------------------|
| ``a == b``    | ``a`` equal to ``b``              || ``a != b``    | ``a`` not equal to ``b``             |
| ``a < b``     | ``a`` less than ``b``             || ``a > b``     | ``a`` greater than ``b``             |
| ``a <= b``    | ``a`` less than or equal to ``b`` || ``a >= b``    | ``a`` greater than or equal to ``b`` |

These comparison operators can be combined with the arithmetic and bitwise operators to express a virtually limitless range of tests for the numbers.
For example, we can check if a number is odd by checking that the modulus with 2 returns 1:

## Identity and Membership Operators 🤖 

Like ``and``, ``or``, and ``not``, Python also contains prose-like operators  to check for identity and membership.
They are the following:

| Operator      | Description                                       |
|---------------|---------------------------------------------------|
| ``a is b``    | True if ``a`` and ``b`` are identical objects     |
| ``a is not b``| True if ``a`` and ``b`` are not identical objects |
| ``a in b``    | True if ``a`` is a member of ``b``                |
| ``a not in b``| True if ``a`` is not a member of ``b``            |


{% highlight python %}
a = 3
b = 4
a is b
{% endhighlight %}




    >>> False




{% highlight python %}
a = 6
b = [123,13,6,27,97]
a in b
{% endhighlight %}




    >>> True



## Data Structures

We have seen Python's simple types: ``int``, ``float``, ``complex``, ``bool``, ``str``, and so on.
Python also has several built-in compound types, which act as containers for other types.
These compound types are:

| Type Name | Example                   |Description                            |
|-----------|---------------------------|---------------------------------------|
| ``list``  | ``[1, 2, 3]``             | Ordered collection                    |
| ``tuple`` | ``(1, 2, 3)``             | Immutable ordered collection          |
| ``dict``  | ``{'a':1, 'b':2, 'c':3}`` | Unordered (key,value) mapping         |
| ``set``   | ``{1, 2, 3}``             | Unordered collection of unique values |

As you can see, round, square, and curly brackets have distinct meanings when it comes to the type of collection produced.
We'll take a quick tour of these data structures here.

## Lists

A list is a collection of items in a particular order. They are defined by comma separated values grouped between square brakets. They can contain objects of any type, even if they are mixed together.


{% highlight python %}
myList = [23,22,22,42,43,"a"]
print(myList)
{% endhighlight %}

    >>> [23, 22, 22, 42, 43, 'a']



{% highlight python %}
myList.append(77)
print(myList)
{% endhighlight %}

    >>> [23, 22, 22, 42, 43, 'a', 77]



{% highlight python %}
myList.pop(0) # removes the elements at position 0
{% endhighlight %}




    >>> 23




{% highlight python %}
myList.count(22) # Returns the number of occurrences of the passed value
{% endhighlight %}




    >>> 2




{% highlight python %}
myList.remove(22) # removes by value
print(myList)
{% endhighlight %}

    >>> [22, 42, 43, 'a', 77]



{% highlight python %}
myList.reverse()
print(myList)
{% endhighlight %}

    >>> [77, 'a', 43, 42, 22]



{% highlight python %}
newList = [1,3,5,78,124,5,23,67,35,7,23,6,12,97]
newList.sort()
print(newList)
{% endhighlight %}

    >>> [1, 3, 5, 5, 6, 7, 12, 23, 23, 35, 67, 78, 97, 124]



{% highlight python %}
# instert at position 0
newList.insert(0,'banana')
print(newList)
{% endhighlight %}

    >>> ['banana', 1, 3, 5, 5, 6, 7, 12, 23, 23, 35, 67, 78, 97, 124]



{% highlight python %}
del newList[0]
print(newList)
{% endhighlight %}

    >>> [1, 3, 5, 5, 6, 7, 12, 23, 23, 35, 67, 78, 97, 124]



{% highlight python %}
#Concatenating two strings
myList + newList
{% endhighlight %}




    >>> [77, 'a', 43, 42, 22, 1, 3, 5, 5, 6, 7, 12, 23, 23, 35, 67, 78, 97, 124]



## Lists: indexing and slicing

Now that we know how lists work, let's focus our energies to index and slice them❗


{% highlight python %}
a = [3,2,5,7,4,2,5,7,1,9]
{% endhighlight %}


{% highlight python %}
# start from the left
# we start counting from 0
a[0] 
{% endhighlight %}




    >>> 3




{% highlight python %}
# start from the righ 
# we start counting from 1
a[-1] 

{% endhighlight %}




    >>> 9




{% highlight python %}
a[-2]
{% endhighlight %}




    >>> 1




{% highlight python %}
a[:6] # from the beginning until the 5th element 🌎
{% endhighlight %}




    >>> [3, 2, 5, 7, 4, 2]




{% highlight python %}
a[-3:] #from the end until the third element from the right
{% endhighlight %}




    >>> [7, 1, 9]



Finally, it is possible to specify a third integer that represents the step size; for example, to select every second element of the list, we can write:



{% highlight python %}
a[::2] # every second element starting from 0
{% endhighlight %}




    >>> [3, 5, 4, 5, 1]




{% highlight python %}
a[1::2] # every second element starting from 1
{% endhighlight %}




    >>> [2, 7, 2, 7, 9]




{% highlight python %}
a[::-1] # reversing the list
{% endhighlight %}




    >>> [9, 1, 7, 5, 2, 4, 7, 5, 2, 3]



## Tuples

Tuples are similar to lists, but they are immutable. 


{% highlight python %}
a = (1,3,5,6,4,2)
print(a)
{% endhighlight %}

    >>> (1, 3, 5, 6, 4, 2)



{% highlight python %}
a[2]
{% endhighlight %}




    >>> 5




{% highlight python %}
a[2] = 3
{% endhighlight %}


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-106-ce05f827d6de> in <module>()
    ----> 1 a[2] = 3
    

    TypeError: 'tuple' object does not support item assignment


Tuples are often used in a Python program; a particularly common case is in functions that have multiple return values. 

The indexing and slicing logic covered earlier for lists works for tuples as well, along with a host of other methods. Refer to the online Python documentation for a more complete list of these.

## Dictionaries
Dictionaries are extremely flexible mappings of keys to values, and form the basis of much of Python's internal implementation.
They can be created via a comma-separated list of ``key:value`` pairs within curly braces:


{% highlight python %}
motorbikes = {'moto-1':'honda🏍️', 'moto-2':'yamaha🏍️', 'moto-3':'suzuki🏍️'}
print(motorbikes)
{% endhighlight %}

    {'moto-1': 'honda🏍️', 'moto-2': 'yamaha🏍️', 'moto-3': 'suzuki🏍️'}



{% highlight python %}
motorbikes['moto-1']
{% endhighlight %}




    >>> 'honda🏍️'




{% highlight python %}
motorbikes['moto-2'] = 'ducati🏍️'
print(motorbikes)
{% endhighlight %}

    >>> {'moto-1': 'honda🏍️', 'moto-2': 'ducati🏍️', 'moto-3': 'suzuki🏍️'}


Dictionaries do not maintain any sense of order for the input parameters. This lack of ordering allows dictionaries to be implemented very efficiently, so that random element access is very fast, regardless of the size of the dictionary (if you're curious how this works, read about the concept of a hash table). The python documentation has a complete list of the methods available for dictionaries.

## Sets

They are defined much like lists and tuples, except they use the curly brackets of dictionaries:


{% highlight python %}
firstTen = {1,2,3,4,5,6,7,8,9,10}
even = {10,12,14,16,18,20}
{% endhighlight %}


{% highlight python %}
firstTen.union(even)
{% endhighlight %}




    >>> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20}




{% highlight python %}
firstTen.intersection(even)
{% endhighlight %}




    >>> {10}




{% highlight python %}
firstTen.difference(even)
{% endhighlight %}


    >>> {1, 2, 3, 4, 5, 6, 7, 8, 9}



Many more set methods and operations are available.
You've probably already guessed what I'll say next: refer to Python's [online documentation](https://docs.python.org/3/library/stdtypes.html) for a complete reference.

## More collections
You can find more info on collections module [here](https://docs.python.org/3/library/collections.html)

In particular, I've found the following very useful on occasion:

- ``collections.namedtuple``: Like a tuple, but each value has a name
- ``collections.defaultdict``: Like a dictionary, but unspecified keys have a user-specified default value
- ``collections.OrderedDict``: Like a dictionary, but the order of keys is maintained

## Loops


{% highlight python %}
## TODO write examples
{% endhighlight %}

## Functions

In Python, functions are defined with the ``def`` statement. They contain code that can be called using the functions name, and take some parameters in input that they can use internally.



{% highlight python %}
def fibonacci(N):
    L = []
    a, b = 0, 1
    while len(L) < N:
        a, b = b, a + b
        L.append(a)
    return L
{% endhighlight %}


{% highlight python %}
fibonacci(10)
{% endhighlight %}




    >>> [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]




{% highlight python %}
def fibonacciMax(max):
    L = []
    a,b = 0,1
    while True:
        a, b = b, a + b
        if a > max:
            break
        L.append(a)
    return L
{% endhighlight %}


{% highlight python %}
fibonacciMax(20)
{% endhighlight %}




    >>> [1, 1, 2, 3, 5, 8, 13]



Functions can take also default arguments:


{% highlight python %}
def fibonacciMaxIn(max, a = 0, b = 1):
    L = []
    while True:
        a, b = b, a + b
        if a > max:
            break
        L.append(a)
    return L
{% endhighlight %}


{% highlight python %}
fibonacciMaxIn(40,2,4)
{% endhighlight %}




    [>>> 4, 6, 10, 16, 26]



## ``*args`` and ``**kwargs``: Flexible Arguments
Sometimes you might wish to write a function in which you don't initially know how many arguments the user will pass.
In this case, you can use the special form ``*args`` and ``**kwargs`` to catch all arguments that are passed.
Here is an example:


{% highlight python %}
def catch_all(*args, **kwargs):
    print("args =", args)
    print("kwargs = ", kwargs)
{% endhighlight %}


{% highlight python %}
catch_all(1, '2', param3=[1,2,3,4], param4='monday')
{% endhighlight %}

    >>> args = (1, '2')
    >>> kwargs =  {'param3': [1, 2, 3, 4], 'param4': 'monday'}


## Anonymous (``lambda``) Functions

These type of function are used when we want to write a function in a compact form


{% highlight python %}
def subtract(a, b):
    return a - b

subtract(4,2)
{% endhighlight %}




    >>> 2



Can be rewritten in its short lambda form


{% highlight python %}
sub = lambda a, b: a - b
sub(6, 3)
{% endhighlight %}




    >>> 3



`def subtract(a, b):` becomes `lambda a, b:` and `return a - b` becomes `a - b`.

This is useful if we are writing functions that are passed as arguments to functions, because it guarantees readability.

## Errors and Exceptions


{% highlight python %}
# TODO
{% endhighlight %}

## Iterators




{% highlight python %}
# TODO
{% endhighlight %}

## List Comprehensions


{% highlight python %}
# TODO
{% endhighlight %}

## Generators


{% highlight python %}
# TODO
{% endhighlight %}

## Strings and Regular Expressions


{% highlight python %}
# TODO
{% endhighlight %}

## Combining Python Data Structures


{% highlight python %}
# TODO
{% endhighlight %}

## Bash and tools in Python
