"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

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


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element."""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a function to corresponding elements."""

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce an iterable from the given initial value."""

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for x in ls:
            val = fn(val, x)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate every element of a list."""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements of two lists."""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum the elements of an iterable."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list."""
    return reduce(mul, 1.0)(ls)
