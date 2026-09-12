"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y


def id(x: float) -> float:
    """$f(x) = x$"""
    return x


def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y


def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -x


def lt(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is less than y else 0.0"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is equal to y else 0.0"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """$f(x) =$ x if x is greater than y else y"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """$f(x) = |x - y| < 1e-2$"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """$f(x) = 1.0 / (1.0 + e^{-x})$"""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """$f(x) =$ x if x is greater than 0, else 0"""
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    """$f(x) = log(x)$"""
    return math.log(x)


def exp(x: float) -> float:
    """$f(x) = e^{x}$"""
    return math.exp(x)


def inv(x: float) -> float:
    """$f(x) = 1/x$"""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """If $f = log$, compute $d * f'(x)$"""
    return d / x


def inv_back(x: float, d: float) -> float:
    """If $f(x) = 1/x$, compute $d * f'(x)$"""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """If $f = relu$, compute $d * f'(x)$"""
    return d if x > 0.0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements of two lists."""
    raise NotImplementedError("Need to implement for Task 0.3")


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate every element of a list."""
    raise NotImplementedError("Need to implement for Task 0.3")


def prod(ls: Iterable[float]) -> float:
    """Product of a list."""
    raise NotImplementedError("Need to implement for Task 0.3")
