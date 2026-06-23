"""
aqc/alpha/alpha_factory.py
============================
Config-driven alpha instantiation factory.

Given a configuration dictionary the factory creates, configures, and
optionally trains alpha instances.  This is the single entry point for
producing alphas in tournament, deployment, and research workflows.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from aqc.alpha.alpha_base import AlphaBase
from aqc.alpha.alpha_registry import AlphaRegistry

logger = logging.getLogger(__name__)


class AlphaFactory:
    """Factory that creates alpha instances from configuration.

    The factory resolves alpha names through the :class:`AlphaRegistry`,
    instantiates them with the supplied parameters, and optionally runs
    a training step.

    Examples
    --------
    >>> factory = AlphaFactory()
    >>> alpha = factory.create("OrderBookImbalanceAlpha", window=20)
    >>> alphas = factory.create_all({
    ...     "OrderBookImbalanceAlpha": {"window": 20},
    ...     "MicropriceAlpha": {"threshold": 0.001},
    ... })
    """

    def __init__(self) -> None:
        self._created: list[AlphaBase] = []

    def create(
        self,
        name: str,
        train_data: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> AlphaBase:
        """Create a single alpha instance.

        Parameters
        ----------
        name:
            Registered alpha name.
        train_data:
            Optional training data.  If provided, ``alpha.fit(train_data)``
            is called after instantiation.
        **kwargs:
            Constructor keyword arguments forwarded to the alpha class.

        Returns
        -------
        AlphaBase
            Configured (and optionally trained) alpha instance.

        Raises
        ------
        KeyError:
            If *name* is not found in the registry.
        TypeError:
            If the registered class cannot be instantiated with *kwargs*.
        """
        alpha_cls = AlphaRegistry.get(name)

        try:
            alpha = alpha_cls(name=name, **kwargs)
        except TypeError as exc:
            logger.error(
                "Failed to instantiate %s with params %s: %s",
                name, kwargs, exc,
            )
            raise

        if train_data is not None:
            logger.info("Training alpha %s on %d rows.", name, len(train_data))
            alpha.fit(train_data)

        self._created.append(alpha)
        logger.info(
            "Alpha created: %s (version=%s, fitted=%s)",
            alpha.name, alpha.version, alpha.is_fitted,
        )
        return alpha

    def create_all(
        self,
        config: dict[str, dict[str, Any]],
        train_data: Optional[pd.DataFrame] = None,
    ) -> list[AlphaBase]:
        """Create multiple alphas from a configuration dictionary.

        Parameters
        ----------
        config:
            ``{alpha_name: {param: value, ...}, ...}``
        train_data:
            Optional training data passed to each alpha's ``fit`` method.

        Returns
        -------
        list[AlphaBase]
            List of created alpha instances.
        """
        alphas: list[AlphaBase] = []
        for name, params in config.items():
            try:
                alpha = self.create(name, train_data=train_data, **params)
                alphas.append(alpha)
            except (KeyError, TypeError) as exc:
                logger.error("Skipping alpha %s: %s", name, exc)
        logger.info(
            "AlphaFactory: created %d/%d alphas.",
            len(alphas), len(config),
        )
        return alphas

    def create_by_category(
        self,
        category: "AlphaCategory",
        train_data: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> list[AlphaBase]:
        """Create all alphas in a given category.

        Parameters
        ----------
        category:
            Alpha category to filter by.
        train_data:
            Optional training data.
        **kwargs:
            Shared constructor arguments.

        Returns
        -------
        list[AlphaBase]
        """
        from aqc.alpha.alpha_base import AlphaCategory

        names = AlphaRegistry.list_by_category(category)
        return [
            self.create(name, train_data=train_data, **kwargs)
            for name in names
        ]

    @property
    def created_alphas(self) -> list[AlphaBase]:
        """All alpha instances created by this factory."""
        return list(self._created)

    @property
    def created_count(self) -> int:
        """Number of alphas created."""
        return len(self._created)

    def clear(self) -> None:
        """Reset the factory's creation history."""
        self._created.clear()
