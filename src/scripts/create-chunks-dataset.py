"""
We want to be able to look at the intervals of MAG data near Mercury across all spacecraft from a statistical point of view.

We define a set of 'near Mercury' conditions.

For each spacecraft:
    First we use SPICE to determine the times when the spacecraft are near Mercury.

    For each time interval
        Then we load data, determine features for that time window, and write as a row in a csv
"""

import datetime as dt
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, cast

import astropy.units as u
import numpy as np
import polars as pl
import pycatch22
import spiceypy as spice
from astropy.coordinates import CartesianDifferential, SkyCoord
from astropy.time import Time
from astropy.wcs.utils import CartesianRepresentation
from hermpy.net import ClientSPICE
from numpy.typing import NDArray
from sunpy.coordinates import frames
from sunpy.time import TimeRange
from tqdm import tqdm

from mvswim.data import get_helios_data, get_parker_data, get_solar_orbiter_data

SAVE_PATH: Path = Path(__file__).parent.parent.parent / "data/data-segments.json"

MAX_DATE: str = "2026-01-01"
FILTER_RESOLUTION: u.Quantity = 1 * u.hour

# Filter options
DISTANCE_FILTER: Tuple[u.Quantity, u.Quantity] = (
    0.3075 * u.au,  # min
    0.4667 * u.au,  # max
)
LATITUDE_FILTER: u.Quantity = 3.38 * u.deg  # +/- value


def main() -> None:

    # The goal is to have a resulting list of data segments which we can turn
    # into a dataset format.
    data_segments: List[DataSegment] = []

    # misc. setup
    print("Fetching SPICE kernels, this may take some time.")
    spice_client = ClientSPICE(KERNEL_LOCATIONS=SPICE_KERNELS)

    observers: List[Spacecraft] = [
        Spacecraft(
            name="Solar Orbiter",
            mission_time_range=TimeRange("2020-02-11", MAX_DATE),
            data_loader=partial(
                get_solar_orbiter_data,
                product="mag-rtn-normal-1-minute",
                quality_limit=2,
            ),
        ),
        Spacecraft(
            name="Parker Solar Probe",
            mission_time_range=TimeRange("2018-08-13", "2025-11-01"),
            data_loader=partial(get_parker_data, product="psp-fld-l2-mag-rtn-1min"),
        ),
        Spacecraft(
            name="Helios 1",
            mission_time_range=TimeRange("1974-12-11", "1985-09-05"),
            data_loader=partial(
                get_helios_data, spacecraft=1, product="40sec_mag_plasma"
            ),
        ),
        Spacecraft(
            name="Helios 2",
            mission_time_range=TimeRange("1976-01-16", "1980-03-09"),
            data_loader=partial(
                get_helios_data, spacecraft=2, product="40sec_mag_plasma"
            ),
        ),
    ]

    with spice_client.KernelPool():

        print("Finding times near-Mercury")

        spacecraft: Spacecraft
        for spacecraft in observers:

            # Get positions for duration of mission
            positions_table: pl.DataFrame = get_positions(spacecraft)

            # Filter only to times within critera
            positions_table = positions_table.filter(
                (pl.col("Distance [au]").is_between(*DISTANCE_FILTER))
                & (pl.col(["Latitude [deg]"]).abs() <= LATITUDE_FILTER)
            )

            # Now we have one large table with all times (at the
            # FILTER_RESOLUTION) when this spacecraft was near Mercury. We are
            # interested in characterising each of these invidiually - and
            # hence want to split them up into individual chunks of time.

            # Determine the jump in time beteen successive rows.
            positions_table = positions_table.with_columns(
                (pl.col("UTC") - pl.col("UTC").shift(1)).alias("Time Step")
            )

            # Timestamps from spice may not be exactly the FILTER RESOLUTION
            # apart. We need to add a small tolerance.
            expected_step = dt.timedelta(hours=FILTER_RESOLUTION.to(u.hour).value)
            tolerance = dt.timedelta(minutes=1)

            jump_indices = np.where(
                positions_table["Time Step"] > expected_step + tolerance
            )[0]

            start_indices = np.concatenate(([0], jump_indices)).tolist()
            end_indices = np.concatenate(
                (jump_indices - 1, [len(positions_table) - 1])
            ).tolist()

            for start_index, end_index in zip(start_indices, end_indices):

                segment_slice = positions_table[
                    start_index : end_index + 1
                ]  # slice once, reuse

                this_data_segement = DataSegment(
                    spacecraft,
                    start_time=positions_table["UTC"][start_index],
                    end_time=positions_table["UTC"][end_index],
                    features={
                        # Heliocentric Distance
                        "Mean Distance [au]": cast(
                            float, segment_slice["Distance [au]"].mean()
                        ),
                        # Longitude [deg]
                        "Mean Longitude [deg]": cast(
                            float, segment_slice["Longitude [deg]"].mean()
                        ),
                        # Latitude [deg]
                        "Mean Latitude [deg]": cast(
                            float, segment_slice["Latitude [deg]"].mean()
                        ),
                    },
                )

                # Also store the full positions table for this segment
                this_data_segement.positions = segment_slice

                # Populate the data segments list
                data_segments.append(this_data_segement)

    print("Fetching data")

    segment: DataSegment
    for i, segment in enumerate(
        data_segments[:]
    ):  # iterate over a copy to allow safe removal
        print(f"Processing segment {i+1}/{len(data_segments[:])}")

        # Load the data, pull out some features
        segment.data = segment.spacecraft.data_loader(segment.time_range)

        # We need to catch if there is no data for this time
        if len(segment.data) == 0:
            data_segments.remove(segment)
            continue

        for component in ["|B|", "Br", "Bt", "Bn"]:

            segment.features[component] = {}

            metrics: Dict[str, Callable[[List[float]], float]] = {
                "Mean": np.mean,
                "Median": np.median,
                "Standard Deviation": np.std,
            }

            for metric_label, metric_function in metrics.items():
                segment.features[component].update(
                    {
                        f"{metric_label}": metric_function(
                            segment.data[component + " [nT]"].drop_nans().to_list()
                        )
                    }
                )

            # Use pycatch22 to determine some metrics
            catch22_features = pycatch22.catch22_all(
                segment.data[component + " [nT]"].drop_nans().to_list()
            )
            segment.features[component].update(
                dict(zip(catch22_features["names"], catch22_features["values"]))
            )

    rows: List[Dict[str, Any]] = []
    for segment in tqdm(data_segments, desc="Creating dataset"):

        assert segment.data is not None
        assert segment.positions is not None

        rows.append(
            {
                "Spacecraft": segment.spacecraft.name,
                "Start": segment.start_time,
                "End": segment.end_time,
                "Features": segment.features,
                "Data": segment.data.to_dict(as_series=False),
                "Positions": segment.positions.drop("Time Step").to_dict(
                    as_series=False
                ),
            }
        )

    print(f"Saving to {SAVE_PATH}")
    with open(SAVE_PATH, "w") as f:
        json.dump(rows, f, cls=DatetimeEncoder, allow_nan=False)


@dataclass
class Spacecraft:
    name: str
    mission_time_range: TimeRange
    data_loader: Callable[[TimeRange], pl.DataFrame]


@dataclass
class DataSegment:
    spacecraft: Spacecraft
    start_time: dt.datetime
    end_time: dt.datetime
    features: Dict[str, Any]
    data: None | pl.DataFrame = None
    positions: None | pl.DataFrame = None

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(self.start_time, self.end_time)

    def __repr__(self) -> str:
        return (
            f"Data from {self.spacecraft.name} spanning:\n"
            + f"    {self.start_time} to {self.end_time}\n"
            + f"    with {len(self.features)} features."
        )


def get_positions(spacecraft: Spacecraft) -> pl.DataFrame:
    """
    Get positions in R, Long., Lat.. Requires furnished spice kernel
    """

    s = spacecraft

    times: List[dt.datetime] = [
        t.center.to_datetime()
        for t in s.mission_time_range.split(
            int(
                (Time(s.mission_time_range.end) - Time(s.mission_time_range.start))
                / FILTER_RESOLUTION
            )
        )
    ]

    ets = spice.datetime2et(times)
    states: NDArray = np.asarray(spice.spkezr(s.name, ets, "J2000", "NONE", "SUN")[0])

    # SPICE gives results in units of km and km/s
    positions = states[:, :3] * u.km
    velocities = states[:, 3:] * (u.km / u.s)

    skycoords = SkyCoord(
        CartesianRepresentation(
            positions.T, differentials=CartesianDifferential(velocities.T)
        ),
        obstime=times,
        frame="icrs",
    ).transform_to(frames.HeliographicStonyhurst)

    assert isinstance(skycoords.radius, u.Quantity)
    assert isinstance(skycoords.lon, u.Quantity)
    assert isinstance(skycoords.lat, u.Quantity)

    radial_velocity = skycoords.d_radius
    tangential_velocity = (skycoords.d_lon * skycoords.radius).to(
        u.km / u.s, equivalencies=u.dimensionless_angles()
    )

    return pl.DataFrame(
        {
            "UTC": times,
            "Distance [au]": skycoords.radius.to(u.au),
            "Longitude [deg]": skycoords.lon.to(u.deg),
            "Latitude [deg]": skycoords.lat.to(u.deg),
            "Radial Velocity [km/s]": radial_velocity,
            "Tangential Velocity [km/s]": tangential_velocity,
        }
    )


SPICE_KERNELS: Dict[str, Dict[str, Any]] = {
    "Generic (tls)": {
        "BASE": "https://naif.jpl.nasa.gov/pub/naif/",
        "DIRECTORY": "generic_kernels/lsk/",
        "PATTERNS": ["naif????.tls", "latest_leapseconds.tls"],
    },
    "Generic (tpc)": {
        "BASE": "https://naif.jpl.nasa.gov/pub/naif/",
        "DIRECTORY": "generic_kernels/pck/",
        "PATTERNS": ["pck00011.tpc"],
    },
    "Solar Orbiter": {
        "BASE": "http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/",
        "DIRECTORY": "kernels/spk/",
        "PATTERNS": [
            "de421.bsp",
            "solo_ANC_soc-orbit_20200210-20301118_L000_V0_00001_V01.bsp",
        ],
    },
    "Parker Solar Probe": {
        "BASE": "https://spdf.gsfc.nasa.gov/pub/data/psp/",
        "DIRECTORY": "ephemeris/spice/ephemerides/",
        "PATTERNS": ["spp_nom_20180812_20300101_v043_PostV7.bsp"],
    },
    "Helios 1/2": {
        "BASE": "https://naif.jpl.nasa.gov/pub/naif/HELIOS/",
        "DIRECTORY": "kernels/spk/",
        "PATTERNS": [
            "???????_???????_?????_?????.bsp",
            "????????_???????_?????_?????.bsp",
        ],
    },
}


class DatetimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()

        return super().default(obj)

    def iterencode(self, obj, _one_shot=False):
        # Replace NaN/Inf with None before encoding
        return super().iterencode(self._replace_nan(obj), _one_shot)

    def _replace_nan(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._replace_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._replace_nan(v) for v in obj]
        return obj


if __name__ == "__main__":
    main()
