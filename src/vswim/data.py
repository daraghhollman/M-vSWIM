# This defers evaluation of type checking until end which allows the circular
# referencing as done in the classes within.
from __future__ import annotations

import datetime as dt
from typing import Set

import polars as pl


class InputData:

    def __init__(
        self,
        data: pl.DataFrame | None,
        region_id: str | None = None,
        orbit_numbers: int | Set[int] | None = None,
        time_column: str = "UTC",
        metadata: dict[str, str] = {},
    ):
        self.data = data
        self._region_id = region_id
        self.metadata = metadata

        if orbit_numbers is not None:
            self._orbit_numbers: Set[int] | None = (
                orbit_numbers
                if isinstance(orbit_numbers, Set)
                else set([orbit_numbers])
            )

        else:
            self._orbit_numbers = None

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
        if self.data is None:
            return

        # Data must have a valid time column
        if self._time_column not in self.data.columns:
            raise ValueError(f"No valid time column ({self._time_column}) in data.")

        # Data must be sorted
        if not self.data.equals(self.data.sort(by=self._time_column)):
            raise ValueError("Data is not chronological.")

    def __add__(self, other: InputData) -> InputData:

        # Check if both objects already have data stored
        if self.data is None or other.data is None:
            raise ValueError("Cannot add objects without associated data.")

        new_data = pl.concat([self.data, other.data])

        return InputData(new_data)

    def __repr__(self) -> str:
        return self.data.__repr__() + "\n" + self.metadata.__repr__()
