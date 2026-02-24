# This defers evaluation of type checking until end which allows the circular
# referencing as done in the classes within.
from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import polars as pl


class MAGData:

    def __init__(
        self,
        data: pl.DataFrame,
        time_column: str = "UTC",
        metadata: dict[str, str] = {},
    ):
        self.data = data
        self.metadata = metadata

        self._time_column = time_column

        self._test_data_format()

    @property
    def length(self) -> dt.timedelta:
        if self.data is None:
            raise ValueError("Can't find the length of data before any is added!")

        data_start = self.data[self._time_column][0]
        data_end = self.data[self._time_column][-1]

        return data_end - data_start

    @property
    def n_rows(self) -> int:
        if self.data is None:
            raise ValueError("Can't find the length of data before any is added!")

        return len(self.data)

    def _test_data_format(self) -> None:

        # Data must have a valid time column
        if self._time_column not in self.data.columns:
            raise ValueError(f"No valid time column ({self._time_column}) in data.")

        # Data must be sorted
        if not self.data.equals(self.data.sort(by=self._time_column)):
            raise ValueError("Data is not chronological.")

    def quickplot(self) -> None:

        _, ax = plt.subplots()

        for column in self.data.columns:

            if column == self._time_column:
                continue

            ax.plot(self.data[self._time_column], self.data[column], label=column)

        ax.legend()

        plt.show()

    def __add__(self, other: MAGData) -> MAGData:

        new_data = pl.concat([self.data, other.data])

        # Merge operator: |
        # Prefers the other in this instance if both define the same metadata.
        new_meta = self.metadata | other.metadata

        return MAGData(new_data, metadata=new_meta)

    def __repr__(self) -> str:
        return self.data.__repr__() + "\n" + self.metadata.__repr__()
