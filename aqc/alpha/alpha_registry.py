"""
aqc/alpha/alpha_registry.py
=============================
Singleton Alpha Registry for the AQC platform.

All alpha classes register themselves here via the ``@register_alpha``
decorator.  The tournament, factory, and monitoring systems use the
registry to discover and instantiate alphas dynamically.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from aqc.alpha.alpha_base import AlphaBase, AlphaCategory

logger = logging.getLogger(__name__)


class AlphaRegistry:
    """Singleton registry for all AQC alpha implementations.

    The registry maps alpha names to their classes.  Alphas register
    themselves at import time using the :func:`register_alpha` decorator.

    Examples
    --------
    >>> @register_alpha
    ... class MyAlpha(AlphaBase):
    ...     ...
    >>> AlphaRegistry.get("MyAlpha")
    <class 'MyAlpha'>
    """

    _instance: Optional["AlphaRegistry"] = None
    _registry: dict[str, Type[AlphaBase]] = {}
    _categories: dict[str, AlphaCategory] = {}

    def __new__(cls) -> "AlphaRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        alpha_cls: Type[AlphaBase],
        name: Optional[str] = None,
        category: AlphaCategory = AlphaCategory.CUSTOM,
    ) -> Type[AlphaBase]:
        """Register an alpha class.

        Parameters
        ----------
        alpha_cls:
            The alpha class to register.
        name:
            Override name (defaults to class name).
        category:
            Alpha category for filtering.

        Returns
        -------
        Type[AlphaBase]
            The same class, unmodified (allows use as decorator).
        """
        reg_name = name or alpha_cls.__name__
        if reg_name in cls._registry:
            logger.warning(
                "Alpha %r already registered — overwriting.", reg_name
            )
        cls._registry[reg_name] = alpha_cls
        cls._categories[reg_name] = category
        logger.info("Alpha registered: %s (category=%s)", reg_name, category.value)
        return alpha_cls

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> Type[AlphaBase]:
        """Retrieve a registered alpha class by name.

        Parameters
        ----------
        name:
            Registered alpha name.

        Returns
        -------
        Type[AlphaBase]

        Raises
        ------
        KeyError:
            If the name is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"Alpha {name!r} not found in registry.  "
                f"Available: [{available}]"
            )
        return cls._registry[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """Return sorted list of all registered alpha names.

        Returns
        -------
        list[str]
        """
        return sorted(cls._registry.keys())

    @classmethod
    def list_by_category(cls, category: AlphaCategory) -> list[str]:
        """Return alpha names filtered by category.

        Parameters
        ----------
        category:
            Category to filter by.

        Returns
        -------
        list[str]
        """
        return sorted(
            name
            for name, cat in cls._categories.items()
            if cat == category
        )

    @classmethod
    def get_all_classes(cls) -> dict[str, Type[AlphaBase]]:
        """Return the full registry mapping.

        Returns
        -------
        dict[str, Type[AlphaBase]]
        """
        return dict(cls._registry)

    @classmethod
    def get_category(cls, name: str) -> AlphaCategory:
        """Return the category of a registered alpha.

        Parameters
        ----------
        name:
            Registered alpha name.

        Returns
        -------
        AlphaCategory
        """
        return cls._categories[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check whether an alpha is registered.

        Parameters
        ----------
        name:
            Alpha name.

        Returns
        -------
        bool
        """
        return name in cls._registry

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove an alpha from the registry.

        Parameters
        ----------
        name:
            Alpha name to remove.
        """
        cls._registry.pop(name, None)
        cls._categories.pop(name, None)
        logger.info("Alpha unregistered: %s", name)

    @classmethod
    def clear(cls) -> None:
        """Remove all registered alphas.  Use in tests only."""
        cls._registry.clear()
        cls._categories.clear()

    @classmethod
    def count(cls) -> int:
        """Number of registered alphas."""
        return len(cls._registry)

    @classmethod
    def summary(cls) -> str:
        """Human-readable summary of the registry."""
        lines = [f"AlphaRegistry: {cls.count()} alpha(s) registered"]
        for name in cls.list_all():
            cat = cls._categories.get(name, AlphaCategory.CUSTOM)
            lines.append(f"  • {name} [{cat.value}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def register_alpha(
    cls: Optional[Type[AlphaBase]] = None,
    *,
    name: Optional[str] = None,
    category: AlphaCategory = AlphaCategory.CUSTOM,
):
    """Decorator to register an alpha class with the global registry.

    Can be used with or without arguments:

    >>> @register_alpha
    ... class MyAlpha(AlphaBase): ...

    >>> @register_alpha(name="custom_name", category=AlphaCategory.ORDERBOOK)
    ... class OBAlpha(AlphaBase): ...

    Parameters
    ----------
    cls:
        The class (when used without arguments).
    name:
        Override registration name.
    category:
        Alpha category.
    """
    def decorator(alpha_cls: Type[AlphaBase]) -> Type[AlphaBase]:
        return AlphaRegistry.register(alpha_cls, name=name, category=category)

    if cls is not None:
        # @register_alpha without arguments
        return AlphaRegistry.register(cls, name=name, category=category)

    # @register_alpha(...) with arguments
    return decorator
