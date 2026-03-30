from __future__ import annotations

import datetime as dt
import random
from typing import Callable

import polars as pl

__all__ = ["GapGenerator"]


class GapGenerator:
    """
    Inserts artificial time-based gaps into a Polars DataFrame.

    Parameters
    ----------
    get_gap_size : Callable[[], float]
        Returns the duration (in minutes) of each gap when called.

    get_gap_interval : Callable[[], float]
        Returns the duration (in minutes) between the end of one gap and the
        start of the next when called.

    seed : int, optional
        Random seed for reproducibility.  Passed to ``random.seed()`` before
        any gap generation so that repeated calls with the same seed always
        produce identical masks.
    """

    def __init__(
        self,
        get_gap_size: Callable[[], float],
        get_gap_interval: Callable[[], float],
        seed: int = 0,
    ) -> None:

        self.get_gap_size = get_gap_size
        self.get_gap_interval = get_gap_interval
        self.seed = seed

    def generate_gaps(
        self,
        df: pl.DataFrame,
        time_column: str = "UTC",
    ) -> pl.DataFrame:
        """
        Return a copy of *df* with rows inside computed gap windows set to
        ``null`` (all non-time columns are nulled; the timestamp column is
        kept so the time axis stays intact).

        The algorithm walks forward through the time axis:
          1. Advance by ``get_gap_interval()`` minutes  → gap starts here.
          2. Advance by ``get_gap_size()`` minutes       → gap ends here.
          3. All rows whose timestamp falls in [gap_start, gap_end) are nulled.
          4. Repeat from the end of the current gap until past the last row.

        Parameters
        ----------
        df : pl.DataFrame
            Source data.  Must contain a datetime column.

        time_column : str
            Name of the UTC timestamp column (default ``"UTC"``). This column
            must be of type: datetime.datetime.

        Returns
        -------
        pl.DataFrame
            DataFrame with the same schema; rows inside gaps have ``null``
            values for every column except *time_column*.
        """

        if df.is_empty():
            raise ValueError("Cannot generate gaps in an empty dataset.")

        # Set seed to ensure reproducibility
        random.seed(self.seed)

        gap_mask = self._build_gap_mask(df[time_column])

        null_series = pl.Series("__gap__", gap_mask)
        data_columns = [c for c in df.columns if c != time_column]

        return df.with_columns(
            [
                pl.when(null_series).then(pl.lit(None)).otherwise(pl.col(c)).alias(c)
                for c in data_columns
            ]
        )

    def train_test_split(
        self,
        df: pl.DataFrame,
        time_column: str = "UTC",
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split *df* into a training set and a test set using gap windows as the
        test segments.

        - **Test set**  — rows that fall inside a gap window, returned with
          their *original* values (no nulls introduced).
        - **Train set** — all remaining rows, also with original values.

        Both DataFrames share the same schema as *df* and the same column
        order.  Row order within each split follows the original time ordering.

        Parameters
        ----------
        df : pl.DataFrame
            Source data.  Must contain a datetime column named *time_column*.

        time_column : str
            Name of the UTC timestamp column (default ``"UTC"``).

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            ``(train_df, test_df)`` — training rows first, test rows second.
        """

        if df.is_empty():
            raise ValueError("Cannot generate gaps in an empty dataset.")

        random.seed(self.seed)
        gap_mask = self._build_gap_mask(df[time_column])

        gap_series = pl.Series("__gap__", gap_mask)
        train_df = df.filter(~gap_series)
        test_df = df.filter(gap_series)

        return train_df, test_df

    def _build_gap_mask(self, times: pl.Series) -> list[bool]:
        """
        Compute a boolean list aligned with *times* where ``True`` marks rows
        that fall inside a gap window.

        Callers are responsible for seeding ``random`` before calling this
        method so that public-method reproducibility guarantees are upheld.

        Parameters
        ----------
        times : pl.Series
            The datetime series to walk over.

        Returns
        -------
        list[bool]
            ``True`` for each row that falls in ``[gap_start, gap_end)``.
        """
        t_min = times.min()
        t_max = times.max()

        assert isinstance(t_min, dt.datetime)
        assert isinstance(t_max, dt.datetime)

        gap_mask: list[bool] = [False] * len(times)
        current_time: dt.datetime = t_min

        while current_time < t_max:
            current_time += dt.timedelta(minutes=self.get_gap_interval())

            if current_time >= t_max:
                break

            gap_start: dt.datetime = current_time
            gap_end: dt.datetime = gap_start + dt.timedelta(minutes=self.get_gap_size())

            in_gap = (times >= gap_start) & (times < gap_end)
            for i, flag in enumerate(in_gap.to_list()):
                if flag:
                    gap_mask[i] = True

            current_time = gap_end

        return gap_mask

    @classmethod
    def from_constants(
        cls,
        gap_size_minutes: float,
        gap_interval_minutes: float,
    ) -> GapGenerator:
        """
        Create a GapGenerator that uses fixed (constant) gap sizes and
        intervals.

        Parameters
        ----------
        gap_size_minutes : float
            Every gap will be exactly this many minutes long.

        gap_interval_minutes : float
            The spacing between gaps will always be exactly this many minutes.
        """
        return cls(
            get_gap_size=lambda: gap_size_minutes,
            get_gap_interval=lambda: gap_interval_minutes,
        )

    @classmethod
    def from_gaussian(
        cls,
        gap_size_mean: float,
        gap_size_std: float,
        gap_interval_mean: float,
        gap_interval_std: float,
        seed: int = 0,
        min_minutes: float = 1.0,
    ) -> GapGenerator:
        """
        Create a GapGenerator whose gap sizes and intervals are drawn from
        independent normal distributions.

        Parameters
        ----------
        gap_size_mean : float
            Mean of the gap-size distribution (minutes).

        gap_size_std : float
            Standard deviation of the gap-size distribution (minutes).

        gap_interval_mean : float
            Mean of the inter-gap interval distribution (minutes).

        gap_interval_std : float
            Standard deviation of the inter-gap interval distribution (minutes).

        seed : int
            Random seed passed to ``random.seed()`` at the start of each
            ``generate_gaps`` call.

        min_minutes : float
            Floor value applied to both drawns to avoid zero or negative
            durations (default 1.0 minute).
        """

        def _get_gap_size() -> float:
            return max(min_minutes, random.gauss(gap_size_mean, gap_size_std))

        def _get_gap_interval() -> float:
            return max(min_minutes, random.gauss(gap_interval_mean, gap_interval_std))

        return cls(
            get_gap_size=_get_gap_size,
            get_gap_interval=_get_gap_interval,
            seed=seed,
        )
