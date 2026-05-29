"""
aqc/research/parameter_space.py
================================
Parameter space definitions for strategy optimisation.

This module provides a clean, type-safe API for declaring the hyperparameter
search space of any AQC strategy.  The space is consumed by
:class:`~aqc.research.optimizer.GridSearchOptimizer` and
:class:`~aqc.research.optimizer.RandomSearchOptimizer`.

Design Goals
------------
* Declarative — express what to search, not how.
* Type-safe — separate classes for int, float, and categorical params.
* Extensible — custom parameter types can be added by subclassing
  :class:`BaseParam`.

Examples
--------
Define the search space for an SMA Crossover strategy::

    space = ParameterSpace()
    space.add(IntParam("fast_period", low=5, high=30, step=5))
    space.add(IntParam("slow_period", low=20, high=100, step=10))

    grid = ParameterGrid(space)
    for params in grid:
        print(params)  # {"fast_period": 5, "slow_period": 20}, ...

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseParam(ABC):
    """Abstract base class for a single hyperparameter dimension.

    Parameters
    ----------
    name:
        Parameter name (must match the keyword argument of the strategy
        constructor or ``params`` dict).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def values(self) -> list[Any]:
        """Return all discrete candidate values for this parameter.

        Returns
        -------
        list[Any]
            Ordered list of candidate values.
        """

    @abstractmethod
    def sample(self, rng: random.Random) -> Any:
        """Draw one random candidate value.

        Parameters
        ----------
        rng:
            :class:`random.Random` instance for reproducible sampling.

        Returns
        -------
        Any
            A single sampled value.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Concrete parameter types
# ---------------------------------------------------------------------------


class IntParam(BaseParam):
    """Integer hyperparameter.

    Produces all integers in ``[low, high]`` with the given *step*.

    Parameters
    ----------
    name:
        Parameter name.
    low:
        Minimum value (inclusive).
    high:
        Maximum value (inclusive).
    step:
        Step size between candidates (default 1).

    Examples
    --------
    >>> p = IntParam("fast_period", low=5, high=30, step=5)
    >>> p.values()
    [5, 10, 15, 20, 25, 30]
    """

    def __init__(self, name: str, low: int, high: int, step: int = 1) -> None:
        super().__init__(name)
        if low > high:
            raise ValueError(f"IntParam '{name}': low ({low}) must be <= high ({high})")
        if step <= 0:
            raise ValueError(f"IntParam '{name}': step must be positive, got {step}")
        self.low = low
        self.high = high
        self.step = step

    def values(self) -> list[int]:
        return list(range(self.low, self.high + 1, self.step))

    def sample(self, rng: random.Random) -> int:
        candidates = self.values()
        return rng.choice(candidates)

    def __repr__(self) -> str:
        return f"IntParam(name={self.name!r}, low={self.low}, high={self.high}, step={self.step})"


class FloatParam(BaseParam):
    """Continuous floating-point hyperparameter.

    For grid search, produces ``n_points`` linearly spaced values in
    ``[low, high]``.  For random search, samples uniformly.

    Parameters
    ----------
    name:
        Parameter name.
    low:
        Minimum value (inclusive).
    high:
        Maximum value (inclusive).
    n_points:
        Number of grid points when used in grid search (default 5).

    Examples
    --------
    >>> p = FloatParam("oversold", low=20.0, high=40.0, n_points=5)
    >>> p.values()
    [20.0, 25.0, 30.0, 35.0, 40.0]
    """

    def __init__(
        self, name: str, low: float, high: float, n_points: int = 5
    ) -> None:
        super().__init__(name)
        if low > high:
            raise ValueError(f"FloatParam '{name}': low ({low}) must be <= high ({high})")
        if n_points < 2:
            raise ValueError(f"FloatParam '{name}': n_points must be >= 2, got {n_points}")
        self.low = low
        self.high = high
        self.n_points = n_points

    def values(self) -> list[float]:
        if self.n_points == 1:
            return [self.low]
        step = (self.high - self.low) / (self.n_points - 1)
        return [round(self.low + i * step, 10) for i in range(self.n_points)]

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def __repr__(self) -> str:
        return (
            f"FloatParam(name={self.name!r}, low={self.low}, "
            f"high={self.high}, n_points={self.n_points})"
        )


class CategoricalParam(BaseParam):
    """Categorical hyperparameter (unordered finite set).

    Parameters
    ----------
    name:
        Parameter name.
    choices:
        List of possible values (any type).

    Examples
    --------
    >>> p = CategoricalParam("allow_short", choices=[True, False])
    >>> p.values()
    [True, False]
    """

    def __init__(self, name: str, choices: list[Any]) -> None:
        super().__init__(name)
        if not choices:
            raise ValueError(f"CategoricalParam '{name}': choices must be non-empty")
        self.choices = list(choices)

    def values(self) -> list[Any]:
        return list(self.choices)

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.choices)

    def __repr__(self) -> str:
        return f"CategoricalParam(name={self.name!r}, choices={self.choices})"


# ---------------------------------------------------------------------------
# ParameterSpace
# ---------------------------------------------------------------------------


class ParameterSpace:
    """Container for a collection of :class:`BaseParam` dimensions.

    A ``ParameterSpace`` defines the complete multi-dimensional search space
    for a strategy's hyperparameters.

    Parameters
    ----------
    params:
        Optional list of :class:`BaseParam` instances to initialise with.

    Examples
    --------
    >>> space = ParameterSpace()
    >>> space.add(IntParam("fast_period", 5, 30, step=5))
    >>> space.add(IntParam("slow_period", 20, 100, step=10))
    >>> len(space)
    2
    >>> space.param_names
    ['fast_period', 'slow_period']
    """

    def __init__(self, params: list[BaseParam] | None = None) -> None:
        self._params: dict[str, BaseParam] = {}
        for p in (params or []):
            self.add(p)

    def add(self, param: BaseParam) -> "ParameterSpace":
        """Add a parameter dimension.

        Parameters
        ----------
        param:
            Any :class:`BaseParam` subclass instance.

        Returns
        -------
        ParameterSpace
            ``self`` for method chaining.

        Raises
        ------
        ValueError
            If a parameter with the same name already exists.
        """
        if param.name in self._params:
            raise ValueError(
                f"Parameter '{param.name}' already exists in this space. "
                "Use a unique name per dimension."
            )
        self._params[param.name] = param
        return self

    def remove(self, name: str) -> "ParameterSpace":
        """Remove a parameter by name.

        Parameters
        ----------
        name:
            Parameter name to remove.

        Returns
        -------
        ParameterSpace
        """
        self._params.pop(name, None)
        return self

    @property
    def param_names(self) -> list[str]:
        """Ordered list of parameter names."""
        return list(self._params.keys())

    @property
    def params(self) -> list[BaseParam]:
        """Ordered list of :class:`BaseParam` objects."""
        return list(self._params.values())

    def grid_size(self) -> int:
        """Return the total number of combinations in the full grid.

        Returns
        -------
        int
            Product of ``len(param.values())`` for all parameters.
        """
        size = 1
        for p in self._params.values():
            size *= len(p.values())
        return size

    def sample(self, rng: random.Random | None = None) -> dict[str, Any]:
        """Draw one random parameter combination.

        Parameters
        ----------
        rng:
            Optional :class:`random.Random` instance.  Uses the global
            random state if ``None``.

        Returns
        -------
        dict[str, Any]
            ``{name: value}`` mapping.
        """
        _rng = rng or random.Random()
        return {name: p.sample(_rng) for name, p in self._params.items()}

    def __len__(self) -> int:
        return len(self._params)

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __repr__(self) -> str:
        param_strs = [f"  {p!r}" for p in self._params.values()]
        return "ParameterSpace(\n" + "\n".join(param_strs) + "\n)"


# ---------------------------------------------------------------------------
# ParameterGrid
# ---------------------------------------------------------------------------


class ParameterGrid:
    """Iterable over all combinations in a :class:`ParameterSpace`.

    Generates the full Cartesian product of parameter values, equivalent
    to scikit-learn's ``ParameterGrid`` but native to AQC.

    Parameters
    ----------
    space:
        The search space to enumerate.

    Examples
    --------
    >>> space = ParameterSpace([IntParam("a", 1, 2), IntParam("b", 10, 20, step=10)])
    >>> list(ParameterGrid(space))
    [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}]
    """

    def __init__(self, space: ParameterSpace) -> None:
        self.space = space

    def __iter__(self) -> Iterator[dict[str, Any]]:
        names = self.space.param_names
        value_lists = [p.values() for p in self.space.params]
        for combo in itertools.product(*value_lists):
            yield dict(zip(names, combo))

    def __len__(self) -> int:
        return self.space.grid_size()

    def __repr__(self) -> str:
        return f"ParameterGrid(size={len(self)}, params={self.space.param_names})"
