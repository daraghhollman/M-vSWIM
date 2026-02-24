# This defers evaluation of type checking until end which allows the circular
# referencing as done in the classes within.
from __future__ import annotations

import datetime as dt
from typing import Set

import polars as pl


class Data:

    def __init__(
        self,
        data: pl.DataFrame | None,
        region_id: str,
        orbit_numbers: int | Set[int],
        time_column: str = "UTC",
    ):
        self.data = data
        self._region_id = region_id
        self._orbit_numbers: Set[int] = (
            orbit_numbers if isinstance(orbit_numbers, Set) else set([orbit_numbers])
        )
        self._time_column = time_column

        self._test_region_id()
        self._test_data_format()
        self._flag_multiple_orbits()

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

    @property  # Region ID should be immutable
    def region_id(self) -> str:
        return self._region_id

    @property  # Orbit numbers should be immutable
    def orbit_numbers(self) -> Set[int]:
        return self._orbit_numbers

    @property  # Any flags should be immutable
    def is_across_multiple_orbits(self) -> bool:
        return self._is_across_multiple_orbits

    def _test_region_id(self) -> None:
        """
        Ensure that input region id is correctly formatted.
        """

        region_options = ["SW", "MSh", "MSp"]

        if self._region_id not in region_options:
            raise ValueError(
                f"Input region ID '{self._region_id}' is not one of the valid options: {region_options}"
            )

    def _flag_multiple_orbits(self) -> None:
        if len(self._orbit_numbers) > 1:
            self._is_across_multiple_orbits = True

        else:
            self._is_across_multiple_orbits = False

    def _test_data_format(self) -> None:
        if self.data is None:
            return

        # Data must have a valid time column
        if self._time_column not in self.data.columns:
            raise ValueError(f"No valid time column ({self._time_column}) in data.")

        # Data must be sorted
        if not self.data.equals(self.data.sort(by=self._time_column)):
            raise ValueError("Data is not chronological.")

    def __add__(self, other: Data) -> Data:

        # Check if regions are the same
        if self.region_id != other.region_id:
            raise ValueError(
                f"Cannot add region data with differing IDs: {self.region_id} and {other.region_id}"
            )

        # Check if both objects already have data stored
        if self.data is None or other.data is None:
            raise ValueError("Cannot add objects without associated data.")

        new_data = pl.concat([self.data, other.data])

        orbit_numbers = set([*self.orbit_numbers, *other.orbit_numbers])

        return Data(new_data, self.region_id, orbit_numbers)

    def __repr__(self) -> str:
        print(f"{self.region_id} data across {len(self._orbit_numbers)} orbit(s)")
        return self.data.__repr__()


# Examples
d1 = Data(
    pl.DataFrame(
        {
            "UTC": [dt.datetime(2020, 1, 1), dt.datetime(2021, 1, 1)],
            "Something Else": [0, 5],
        }
    ),
    "SW",
    1,
)

d2 = Data(
    pl.DataFrame(
        {
            "UTC": [dt.datetime(2022, 1, 1), dt.datetime(2023, 1, 1)],
            "Something Else": [0, 5],
        }
    ),
    "SW",
    {2, 5, 18},
)

print(d1 + d2)
