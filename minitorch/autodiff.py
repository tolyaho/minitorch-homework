from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    plus = [v + epsilon if i == arg else v for i, v in enumerate(vals)]
    minus = [v - epsilon if i == arg else v for i, v in enumerate(vals)]
    return (f(*plus) - f(*minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add a derivative contribution to this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable was created by the user."""
        ...

    def is_constant(self) -> bool:
        """True if this variable does not require a derivative."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Inputs used to create this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Return local derivatives paired with the corresponding inputs."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order: List[Variable] = []
    visited = set()

    def visit(var: Variable) -> None:
        if var.is_constant() or var.unique_id in visited:
            return
        visited.add(var.unique_id)
        for parent in var.parents:
            visit(parent)
        order.append(var)

    visit(variable)
    return list(reversed(order))


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Starting derivative to propagate backward to the leaves

    """
    derivatives: Dict[int, Any] = {}
    derivatives[variable.unique_id] = deriv
    for var in topological_sort(variable):
        d = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(d)
        else:
            for parent, parent_deriv in var.chain_rule(d):
                derivatives[parent.unique_id] = (
                    derivatives.get(parent.unique_id, 0.0) + parent_deriv
                )


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Values stored during the forward pass."""
        return self.saved_values
