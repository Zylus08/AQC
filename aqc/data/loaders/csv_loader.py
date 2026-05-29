"""
aqc/data/loaders/csv_loader.py
==============================
CSV-based market data loader.

Supports daily OHLCV files and intraday minute/tick files.  The loader
validates the schema, parses timestamps, handles missing values, and returns
clean :class:`~pandas.DataFrame` objects ready for the backtest engine.

Supported file layouts
----------------------

Daily::

    date,open,high,low,close,volume
    2024-01-02,150.05,152.30,149.80,151.90,45000000

Intraday::

    datetime,open,high,low,close,volume
    2024-01-02 09:30:00,150.00,150.50,149.90,150.20,500000

The loader normalises column names to lowercase and maps common aliases
(e.g. ``Date`` → ``date``, ``Adj Close`` → ``close``).

Author: AQC Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column names required in the output DataFrame
REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

# Maps common raw column names → canonical names
COLUMN_ALIASES: dict[str, str] = {
    "date": "date",
    "datetime": "datetime",
    "timestamp": "datetime",
    "time": "datetime",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "adj close": "close",
    "adj_close": "close",
    "adjusted_close": "close",
    "turnover": "volume",
    "qty": "volume",
}


class DataLoaderError(Exception):
    """Raised when the data loader encounters an unrecoverable error."""


class CSVDataLoader:
    """Load and validate OHLCV data from CSV files.

    Parameters
    ----------
    data_dir:
        Root directory where CSV files are stored.  Individual files are
        resolved relative to this path.
    datetime_column:
        Name of the timestamp column in raw files (before normalisation).
    datetime_format:
        :func:`~pandas.to_datetime` format string.  ``None`` enables
        pandas' automatic parser.
    fill_method:
        Strategy for handling missing OHLCV values.

        ``"ffill"``
            Forward-fill — use the previous bar's value.
        ``"bfill"``
            Backward-fill — use the next bar's value.
        ``"drop"``
            Drop rows with any missing OHLCV values.
        ``None``
            Leave ``NaN`` values in place.
    tz:
        Timezone to localise the DatetimeIndex to.  ``None`` leaves
        timestamps timezone-naive.

    Examples
    --------
    >>> loader = CSVDataLoader(data_dir="data/raw")
    >>> df = loader.load("AAPL.csv")
    >>> df.head()
    """

    def __init__(
        self,
        data_dir: str = ".",
        datetime_column: str = "date",
        datetime_format: Optional[str] = None,
        fill_method: Optional[str] = "ffill",
        tz: Optional[str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.datetime_column = datetime_column
        self.datetime_format = datetime_format
        self.fill_method = fill_method
        self.tz = tz

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, filename: str, symbol: Optional[str] = None) -> pd.DataFrame:
        """Load a single OHLCV CSV file.

        Parameters
        ----------
        filename:
            File name (relative to *data_dir*) or absolute path.
        symbol:
            Optional symbol label used in log messages.

        Returns
        -------
        pd.DataFrame
            Clean OHLCV DataFrame indexed by a :class:`~pandas.DatetimeIndex`.

        Raises
        ------
        DataLoaderError:
            If the file is missing, unreadable, or fails schema validation.
        """
        path = self._resolve_path(filename)
        label = symbol or path.stem

        logger.info("Loading %s from %s", label, path)

        try:
            raw = pd.read_csv(path)
        except FileNotFoundError:
            raise DataLoaderError(f"File not found: {path}")
        except Exception as exc:
            raise DataLoaderError(f"Failed to read {path}: {exc}") from exc

        df = self._normalise_columns(raw)
        df = self._parse_index(df)
        df = self._validate_schema(df, label)
        df = self._handle_missing(df)
        df = self._cast_numeric(df)
        df = self._sort_index(df)

        if self.tz:
            try:
                df.index = df.index.tz_localize(self.tz)
            except TypeError:
                df.index = df.index.tz_convert(self.tz)

        logger.info(
            "Loaded %s: %d bars [%s → %s]",
            label,
            len(df),
            df.index[0],
            df.index[-1],
        )
        return df

    def load_multiple(
        self, files: dict[str, str]
    ) -> dict[str, pd.DataFrame]:
        """Load multiple symbols at once.

        Parameters
        ----------
        files:
            Mapping of ``symbol → filename``.

        Returns
        -------
        dict[str, pd.DataFrame]
            Clean DataFrames keyed by symbol.

        Examples
        --------
        >>> data = loader.load_multiple({"AAPL": "AAPL.csv", "MSFT": "MSFT.csv"})
        """
        result: dict[str, pd.DataFrame] = {}
        for symbol, filename in files.items():
            try:
                result[symbol] = self.load(filename, symbol=symbol)
            except DataLoaderError as exc:
                logger.error("Skipping %s: %s", symbol, exc)
        return result

    def load_intraday(
        self,
        filename: str,
        symbol: Optional[str] = None,
        resample_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load intraday OHLCV data with optional resampling.

        Parameters
        ----------
        filename:
            Intraday CSV file (minute bars or tick data).
        symbol:
            Optional symbol label.
        resample_to:
            Pandas resample rule string (e.g. ``"5T"`` for 5-minute bars,
            ``"1H"`` for hourly).  ``None`` returns raw bars.

        Returns
        -------
        pd.DataFrame
            Clean (and optionally resampled) intraday OHLCV DataFrame.
        """
        df = self.load(filename, symbol=symbol)
        if resample_to:
            df = self._resample(df, resample_to)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, filename: str) -> Path:
        """Return an absolute Path for *filename*.

        Parameters
        ----------
        filename:
            Relative or absolute path string.

        Returns
        -------
        Path
        """
        p = Path(filename)
        if p.is_absolute():
            return p
        return self.data_dir / filename

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case and alias column names.

        Parameters
        ----------
        df:
            Raw DataFrame from :func:`~pandas.read_csv`.

        Returns
        -------
        pd.DataFrame
            DataFrame with normalised column names.
        """
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        rename_map = {
            col: COLUMN_ALIASES[col]
            for col in df.columns
            if col in COLUMN_ALIASES and COLUMN_ALIASES[col] != col
        }
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def _parse_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DataFrame index to a DatetimeIndex.

        Looks for ``date`` or ``datetime`` columns (in that priority order).

        Parameters
        ----------
        df:
            Normalised DataFrame.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        DataLoaderError:
            If no timestamp column can be found.
        """
        ts_col = None
        for candidate in ("datetime", "date"):
            if candidate in df.columns:
                ts_col = candidate
                break

        if ts_col is None:
            # Try to parse the current index
            try:
                df.index = pd.to_datetime(df.index, format=self.datetime_format)
                return df
            except Exception:
                raise DataLoaderError(
                    "No timestamp column found.  Expected 'date' or 'datetime'."
                )

        df[ts_col] = pd.to_datetime(df[ts_col], format=self.datetime_format)
        df = df.set_index(ts_col)
        df.index.name = "timestamp"
        return df

    def _validate_schema(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Check that all required OHLCV columns are present.

        Parameters
        ----------
        df:
            DataFrame with parsed index.
        label:
            Symbol label for error messages.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        DataLoaderError:
            If any required column is missing.
        """
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataLoaderError(
                f"Missing required columns for {label}: {missing}. "
                f"Available: {list(df.columns)}"
            )
        return df[list(REQUIRED_COLUMNS) + [c for c in df.columns if c not in REQUIRED_COLUMNS]]

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured missing-value strategy.

        Parameters
        ----------
        df:
            Validated DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        ohlcv_cols = list(REQUIRED_COLUMNS)
        initial_nulls = df[ohlcv_cols].isnull().sum().sum()

        if initial_nulls == 0:
            return df

        logger.warning(
            "Found %d missing values — applying strategy '%s'",
            initial_nulls,
            self.fill_method,
        )

        if self.fill_method == "ffill":
            df[ohlcv_cols] = df[ohlcv_cols].ffill()
        elif self.fill_method == "bfill":
            df[ohlcv_cols] = df[ohlcv_cols].bfill()
        elif self.fill_method == "drop":
            df = df.dropna(subset=ohlcv_cols)
        # else: leave NaN in place

        return df

    def _cast_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLCV columns are float64.

        Parameters
        ----------
        df:
            DataFrame after missing-value handling.

        Returns
        -------
        pd.DataFrame
        """
        for col in REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _sort_index(df: pd.DataFrame) -> pd.DataFrame:
        """Sort by timestamp ascending.

        Parameters
        ----------
        df:
            DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return df.sort_index()

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV data to a lower frequency.

        Uses standard OHLCV aggregation rules: open=first, high=max,
        low=min, close=last, volume=sum.

        Parameters
        ----------
        df:
            Higher-frequency OHLCV DataFrame.
        rule:
            Pandas resample rule (e.g. ``"5T"``, ``"1H"``).

        Returns
        -------
        pd.DataFrame
        """
        resampled = df.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        return resampled.dropna(subset=["close"])

    def __repr__(self) -> str:
        return f"CSVDataLoader(data_dir={self.data_dir}, fill={self.fill_method})"
