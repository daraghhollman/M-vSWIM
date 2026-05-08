"""
A script showing an overview of the trajectories of multiple spacecraft with respect to Mercury. We will include plots of latitude and velocity as a function of heliocentric distance. We will include the full sections, and times where they are near Mercury.
"""

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import astropy.units as u
import matplotlib.pyplot as plt
import polars as pl
import spiceypy as spice
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.time import Time
from hermpy.net import ClientSPICE
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sunpy.coordinates import frames
from sunpy.time import TimeRange

DATA_SEGMENTS = Path(__file__).parent.parent.parent.parent / "data/data-segments.json"

COLOURS = {
    "Object": {
        "Solar Orbiter": "indianred",
        "Parker Solar Probe": "cornflowerblue",
        "Helios 1": "green",
        "Helios 2": "blue",
    }
}

# These parameters don't really matter for this as its just visualisation
FILTER_RESOLUTION: u.Quantity = 1 * u.hour


def main() -> None:

    # Load data from json. This file contains entries for each time a
    # spacecraft was within certain bounds of Mercury's orbit.
    data_segments: List[DataSegment] = load_segments(DATA_SEGMENTS)

    # 2 axes, latitude vs heliocentric distance, and velocity vs heliocentric distance
    fig: Figure
    axes: List[Axes]
    fig, axes = plt.subplots(1, 2)

    latitude_ax = axes[0]
    velocity_ax = axes[1]

    # LATITUDE
    ax = latitude_ax

    # Plot trajectory for each chunk
    for s in data_segments:
        ax.plot(
            s.positions["Distance [au]"],
            s.positions["Latitude [deg]"],
            color=COLOURS["Object"][s.spacecraft],
        )

    # Add a year of Mercury for context
    mercury = Spacecraft("Mercury", TimeRange("2020-01-01", "2021-01-01"))
    spice_client = ClientSPICE()
    spice_client.KERNEL_LOCATIONS.update(SPICE_KERNELS)
    with spice_client.KernelPool():
        mercury_positions = get_positions(mercury)

    ax.plot(mercury_positions["Distance [au]"], mercury_positions["Latitude [deg]"])

    ax.set(xlabel="Heliocentric Distance [au]", ylabel="Heliographic Latitude [deg]")

    for ax in axes:
        ax.margins(0)

    plt.show()


@dataclass
class DataSegment:
    spacecraft: str
    start_time: dt.datetime
    end_time: dt.datetime
    features: Dict[str, Any]
    data: pl.DataFrame
    positions: pl.DataFrame

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(self.start_time, self.end_time)

    def __repr__(self) -> str:
        return (
            f"Data from {self.spacecraft} spanning:\n"
            + f"    {self.start_time} to {self.end_time}\n"
            + f"    with {len(self.features)} features."
        )


def load_segments(path: Path) -> List[DataSegment]:
    with open(path, "r") as f:
        raw: List[Dict[str, Any]] = json.load(f, object_hook=_decode)

    return [
        DataSegment(
            spacecraft=row["Spacecraft"],
            start_time=row["Start"],
            end_time=row["End"],
            features=row["Features"],
            data=pl.DataFrame(row["Data"]),
            positions=pl.DataFrame(row["Positions"]),
        )
        for row in raw
    ]


def _decode(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _decode(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_decode(item) for item in data]
    if isinstance(data, str):
        try:
            return dt.datetime.fromisoformat(data)
        except ValueError:
            pass
        try:
            return float(data)
        except ValueError:
            pass
    return data


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
}


@dataclass
class Spacecraft:
    name: str
    mission_time_range: TimeRange


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
    positions: NDArray = spice.spkpos(s.name, ets, "J2000", "NONE", "SUN")[0]

    # SPICE gives results in units of km
    positions *= u.km

    skycoords = SkyCoord(
        CartesianRepresentation(positions.T),
        obstime=times,
        frame="icrs",
    ).transform_to(frames.HeliographicStonyhurst)

    assert isinstance(skycoords.radius, u.Quantity)
    assert isinstance(skycoords.lon, u.Quantity)
    assert isinstance(skycoords.lat, u.Quantity)

    return pl.DataFrame(
        {
            "UTC": times,
            "Distance [au]": skycoords.radius.to(u.au),
            "Longitude [deg]": skycoords.lon.to(u.deg),
            "Latitude [deg]": skycoords.lat.to(u.deg),
        }
    )


if __name__ == "__main__":
    main()
