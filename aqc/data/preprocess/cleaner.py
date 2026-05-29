"""
aqc/data/preprocess/cleaner.py
==============================
Data cleaning and preprocessing utilities.

The :class:`DataCleaner` pipeline accepts a raw OHLCV DataFrame and applies
a series of transformations:

1. Remove duplicate timestamps.
2. Remove rows where OHLCV values are non-positive.
3. Detect and handle outliers (price spikes).
4. Detect and handle gaps (missing bars in intraday data).
5. Normalise volume.

Each step can be enabled / disabled via constructor flags, making the
cleaner composable and testable.

Author: AQC Team
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Apply a configurable cleaning pipeline to OHLCV DataFrames.

    Parameters
    ----------
    remove_duplicates:
        Drop rows with duplicate index values (keep last).
    remove_non_positive:
        Drop rows where any of open / high / low / close ≤ 0.
    outlier_std_threshold:
        If set, flag and remove bars where the log-return exceeds this
        many standard deviations (based on a rolling 20-bar window).
        ``None`` disables outlier detection.
    fill_gaps:
        If ``True``, forward-fill any internal timestamp gaps detected
        after resampling to the dominant frequency.

    Examples
    --------
    >>> cleaner = DataCleaner(outlier_std_threshold=5.0)
    >>> clean_df = cleaner.clean(raw_df)
    """

    def __init__(
        self,
        remove_duplicates: bool = True,
        remove_non_positive: bool = True,
        outlier_std_threshold: Optional[float] = 5.0,
        fill_gaps: bool = False,
    ) -> None:
        self.remove_duplicates = remove_duplicates
        self.remove_non_positive = remove_non_positive
        self.outlier_std_threshold = outlier_std_threshold
        self.fill_gaps = fill_gaps

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline.

        Parameters
        ----------
        df:
            Raw OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.
        """
        original_len = len(df)
        df = df.copy()

        if self.remove_duplicates:
            df = self._drop_duplicates(df)

        if self.remove_non_positive:
            df = self._drop_non_positive(df)

        if self.outlier_std_threshold is not None:
            df = self._remove_outliers(df, self.outlier_std_threshold)

        if self.fill_gaps:
            df = self._fill_gaps(df)

        removed = original_len - len(df)
        if removed > 0:
            logger.info("DataCleaner removed %d rows (%d → %d)", removed, original_len, len(df))

        return df

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate index entries, keeping the last occurrence."""
        before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        removed = before - len(df)
        if removed:
            logger.warning("Dropped %d duplicate timestamps", removed)
        return df

    @staticmethod
    def _drop_non_positive(df: pd.DataFrame) -> pd.DataFrame:
        """Drop bars with any non-positive OHLCV price."""
        cols = ["open", "high", "low", "close"]
        price_cols = [c for c in cols if c in df.columns]
        mask = (df[price_cols] > 0).all(axis=1)
        before = len(df)
        df = df[mask]
        removed = before - len(df)
        if removed:
            logger.warning("Dropped %d non-positive price rows", removed)
        return df

    @staticmethod
    def _remove_outliers(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Remove bars where close log-return is an extreme outlier.

        Parameters
        ----------
        df:
            OHLCV DataFrame.
        threshold:
            Number of rolling standard deviations to flag as an outlier.
        """
        if "close" not in df.columns or len(df) < 21:
            return df

        log_returns = np.log(df["close"] / df["close"].shift(1))
        rolling_std = log_returns.rolling(20, min_periods=5).std()
        z_scores = (log_returns - log_returns.rolling(20, min_periods=5).mean()) / rolling_std
        outlier_mask = z_scores.abs() > threshold

        before = len(df)
        df = df[~outlier_mask.fillna(False)]
        removed = before - len(df)
        if removed:
            logger.warning(
                "Removed %d outlier bars (|z| > %.1f)", removed, threshold
            )
        return df

    @staticmethod
    def _fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill gaps detected by resampling to the dominant frequency.

        Parameters
        ----------
        df:
            OHLCV DataFrame with DatetimeIndex.
        """
        if len(df) < 2:
            return df

        # Infer the dominant bar frequency
        deltas = pd.Series(df.index).diff().dropna()
        dominant_freq = deltas.mode()[0]

        expected_index = pd.date_range(
            start=df.index[0], end=df.index[-1], freq=dominant_freq
        )
        df = df.reindex(expected_index)
        gaps = df["close"].isnull().sum()
        if gaps:
            logger.info("Forward-filling %d gap bars", gaps)
            df = df.ffill()

        return df

    def add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append a ``log_return`` column to the DataFrame.

        Parameters
        ----------
        df:
            Clean OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with an additional ``log_return`` column.
        """
        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append a ``return`` column (simple arithmetic returns).

        Parameters
        ----------
        df:
            Clean OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()
        df["return"] = df["close"].pct_change()
        return df

    def __repr__(self) -> str:
        return (
            f"DataCleaner("
            f"dedup={self.remove_duplicates}, "
            f"nonpos={self.remove_non_positive}, "
            f"outlier_std={self.outlier_std_threshold})"
        )
