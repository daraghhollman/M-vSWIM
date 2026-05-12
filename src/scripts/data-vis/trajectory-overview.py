"""
A script showing an overview of the trajectories of multiple spacecraft with
respect to Mercury. We will include plots of latitude and velocity as a
function of heliocentric distance. We will include the full sections, and times
where they are near Mercury.
"""

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import spiceypy as spice
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, SkyCoord
from astropy.time import Time
from hermpy.net import ClientSPICE
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from sunpy.coordinates import frames
from sunpy.time import TimeRange

DATA_SEGMENTS = Path(__file__).parent.parent.parent.parent / "data/data-segments.json"
FIGURE_OUTPUT = (
    Path(__file__).parent.parent.parent.parent / "figures/trajectory-overview.pdf"
)

COLOURS = {
    "Object": {
        "Solar Orbiter": "#648FFF",
        "Parker Solar Probe": "#785EF0",
        "Helios 1": "#DC267F",
        "Helios 2": "#FE6100",
    }
}

# These parameters don't really matter for this as its just visualisation
FILTER_RESOLUTION: u.Quantity = 1 * u.day
MAX_DATE: str = "2026-01-01"


def main() -> None:

    # Make output directory if it doesn't exist
    FIGURE_OUTPUT.parent.mkdir(exist_ok=True)

    # Load data from json. This file contains entries for each time a
    # spacecraft was within certain bounds of Mercury's orbit.
    data_segments: List[DataSegment] = load_segments(DATA_SEGMENTS)

    unique_spacecraft: Set[str] = set([s.spacecraft for s in data_segments])

    # 2 axes, latitude vs heliocentric distance, and velocity vs heliocentric distance
    fig: Figure = plt.figure(figsize=(8, 4))
    gs = GridSpec(2, 2, figure=fig)

    latitude_ax = fig.add_subplot(gs[:, 0])
    radial_velocity_ax = fig.add_subplot(gs[0, 1])
    tangential_velocity_ax = fig.add_subplot(gs[1, 1])

    axes: List[Axes] = [latitude_ax, radial_velocity_ax, tangential_velocity_ax]

    spice_client = ClientSPICE()
    spice_client.KERNEL_LOCATIONS.update(SPICE_KERNELS)
    with spice_client.KernelPool():

        # Add a year of Mercury for context
        mercury = Spacecraft("Mercury", TimeRange("2020-01-01", "2021-01-01"))
        mercury_positions = get_positions(mercury)
        mercury_velocities = get_velocities(mercury)

        # BACKGROUND
        # For each unqiue spacecraft, we want to pull their positions for the
        # entire time for reference.
        for name in unique_spacecraft:

            positions = get_positions(Spacecraft(name, MISSION_TIME_RANGES[name]))
            velocities = get_velocities(Spacecraft(name, MISSION_TIME_RANGES[name]))

            background_params = {
                "zorder": -1,
                "alpha": 0.2,
                "color": COLOURS["Object"][name],
            }

            latitude_ax.plot(
                positions["Distance [au]"],
                positions["Latitude [deg]"],
                **background_params,
            )

            radial_velocity_ax.plot(
                positions["Distance [au]"],
                velocities["Radial Velocity [km/s]"],
                **background_params,
            )

            tangential_velocity_ax.plot(
                positions["Distance [au]"],
                velocities["Tangential Velocity [km/s]"],
                **background_params,
            )

    # CHUNK DATA
    # Plot trajectory for each chunk
    spacecraft_to_plot = unique_spacecraft.copy()
    for s in data_segments:

        if s.spacecraft in spacecraft_to_plot:
            label = s.spacecraft
            spacecraft_to_plot.remove(s.spacecraft)

        else:
            label = ""

        latitude_ax.plot(
            s.positions["Distance [au]"],
            s.positions["Latitude [deg]"],
            color=COLOURS["Object"][s.spacecraft],
            lw=1,
            label=label,
        )
        radial_velocity_ax.plot(
            s.positions["Distance [au]"],
            s.positions["Radial Velocity [km/s]"],
            color=COLOURS["Object"][s.spacecraft],
            lw=1,
            label=label,
        )
        tangential_velocity_ax.plot(
            s.positions["Distance [au]"],
            s.positions["Tangential Velocity [km/s]"],
            color=COLOURS["Object"][s.spacecraft],
            lw=1,
            label=label,
        )

    # ADD MERCURY AS REFERENCE
    latitude_ax.plot(
        mercury_positions["Distance [au]"],
        mercury_positions["Latitude [deg]"],
        color="black",
        lw=3,
        label="Mercury",
    )
    radial_velocity_ax.plot(
        mercury_positions["Distance [au]"],
        mercury_velocities["Radial Velocity [km/s]"],
        color="black",
        lw=3,
        label="Mercury",
    )
    tangential_velocity_ax.plot(
        mercury_positions["Distance [au]"],
        mercury_velocities["Tangential Velocity [km/s]"],
        color="black",
        lw=3,
        label="Mercury",
    )

    latitude_ax.set(
        xlabel="Heliocentric Distance [au]",
        ylabel="Heliographic Latitude [deg]",
        ylim=(-6, 6),
    )
    latitude_ax.legend(ncols=2, loc="lower center", bbox_to_anchor=(1, 1))

    radial_velocity_ax.set(
        ylabel="$v_R$ [km/s]",
        xticks=[],
        ylim=(-60, 60)
    )

    tangential_velocity_ax.set(
        xlabel="Heliocentric Distance [au]",
        ylabel="$v_T$ [km/s]",
        ylim=(0, 70)
    )

    for ax in axes:

        ax.set(
            xlim=(0.2, 1.1),
        )
        ax.margins(0)

    fig.subplots_adjust(left=0.1, right=0.95, top=0.8)
    fig.savefig(FIGURE_OUTPUT, format="pdf")


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


def get_velocities(spacecraft: Spacecraft) -> pl.DataFrame:
    """
    Get velocities in radial and tangential components in the
    HeliographicStonyhurst frame. Requires furnished spice kernel.
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
    # spkezr returns both position and velocity as a 6-vector [x, y, z, vx, vy, vz]
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
            "Radial Velocity [km/s]": radial_velocity,
            "Tangential Velocity [km/s]": tangential_velocity,
        }
    )


# A lookup table of mission time ranges
MISSION_TIME_RANGES: Dict[str, TimeRange] = {
    "Helios 1": TimeRange("1974-12-11", "1985-09-05"),
    "Helios 2": TimeRange("1976-01-16", "1980-03-09"),
    "Solar Orbiter": TimeRange("2020-02-11", MAX_DATE),
    "Parker Solar Probe": TimeRange("2018-08-13", "2025-11-01"),
}


if __name__ == "__main__":
    main()
