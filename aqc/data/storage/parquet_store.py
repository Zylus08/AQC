"""
aqc/data/storage/parquet_store.py
==================================
Parquet-based local data cache.

Provides fast read/write of OHLCV DataFrames using the Parquet columnar
format via :mod:`pyarrow`.  Intended as a transparent cache layer between
the CSV loader and the strategy — avoid re-parsing CSVs on every run.

Author: AQC Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ParquetStore:
    """Read and write OHLCV DataFrames as Parquet files.

    Parameters
    ----------
    store_dir:
        Root directory where Parquet files are stored.

    Examples
    --------
    >>> store = ParquetStore(store_dir="data/cache")
    >>> store.save("AAPL", df)
    >>> df = store.load("AAPL")
    """

    def __init__(self, store_dir: str = "data/cache") -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, symbol: str, df: pd.DataFrame) -> Path:
        """Persist a DataFrame to a Parquet file.

        Parameters
        ----------
        symbol:
            Instrument ticker used as the file name.
        df:
            OHLCV DataFrame to save.

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        path = self._path(symbol)
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        logger.info("Saved %s → %s (%d rows)", symbol, path, len(df))
        return path

    def load(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load a DataFrame from the Parquet cache.

        Parameters
        ----------
        symbol:
            Instrument ticker.

        Returns
        -------
        pd.DataFrame | None
            Cached DataFrame, or ``None`` if the file does not exist.
        """
        path = self._path(symbol)
        if not path.exists():
            logger.debug("Cache miss for %s — file not found: %s", symbol, path)
            return None
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("Loaded %s from cache (%d rows)", symbol, len(df))
        return df

    def exists(self, symbol: str) -> bool:
        """Check whether a cached file exists for *symbol*.

        Parameters
        ----------
        symbol:
            Instrument ticker.

        Returns
        -------
        bool
        """
        return self._path(symbol).exists()

    def delete(self, symbol: str) -> bool:
        """Remove the cached file for *symbol*.

        Parameters
        ----------
        symbol:
            Instrument ticker.

        Returns
        -------
        bool
            ``True`` if the file was deleted, ``False`` if it did not exist.
        """
        path = self._path(symbol)
        if path.exists():
            path.unlink()
            logger.info("Deleted cache for %s", symbol)
            return True
        return False

    def list_cached(self) -> list[str]:
        """Return a list of all cached symbol names.

        Returns
        -------
        list[str]
        """
        return [p.stem for p in self.store_dir.glob("*.parquet")]

    def _path(self, symbol: str) -> Path:
        """Build the Parquet file path for *symbol*.

        Parameters
        ----------
        symbol:
            Instrument ticker.

        Returns
        -------
        Path
        """
        safe_name = symbol.replace("/", "_").replace(":", "_")
        return self.store_dir / f"{safe_name}.parquet"

    def __repr__(self) -> str:
        cached = self.list_cached()
        return f"ParquetStore(dir={self.store_dir}, cached={cached})"
