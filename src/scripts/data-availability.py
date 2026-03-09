"""
Data availability is not the same as spacecraft residence. Just because a
spacecraft is within the orbital range of Mercury, does not mean that the
instruments are on.
"""

import datetime as dt
from pathlib import Path
from typing import Any, Dict

import astropy.units as u
import numpy as np
import polars as pl
import spiceypy as spice
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.units import Quantity
from hermpy.net import ClientSPICE
from sunpy.coordinates import frames
from sunpy.time import TimeRange

from mvswim.data import get_solar_orbiter_data

# Define the critera with which we look within
HELIOCENTRIC_DISTANCE_BOUNDS = (0.3, 0.5)  # au
TRAJECTORY_RESOLUTION = dt.timedelta(days=1)

DATA_DIRECTORY = Path(__file__).parent.parent.parent / "data/"
FIGURE_DIRECTORY = Path(__file__).parent.parent.parent / "figures/"


def main():
    # This hermpy spice client will by default include MESSENGER kernels for the
    # orbital phase of the mission. We must add other kernels to download.
    spice_client = ClientSPICE()

    spice_client.KERNEL_LOCATIONS.update(
        {
            # Solar Orbiter
            "Solar Orbiter Frames": {
                "BASE": "http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/",
                "DIRECTORY": "kernels/fk/",
                "PATTERNS": ["solo_ANC_soc-sci-fk_V09.tf"],
            },
            "Solar Orbiter Positions": {
                "BASE": "http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/",
                "DIRECTORY": "kernels/spk/",
                "PATTERNS": [
                    "de421.bsp",
                    "solo_ANC_soc-orbit_20200210-20301118_L000_V0_00001_V01.bsp",
                ],
            },
        }
    )

    spacecraft_info: Dict[str, Dict[str, Any]] = {
        "Solar Orbiter": {
            "ID": "Solar Orbiter",
            "Time Range": (dt.datetime(2020, 2, 11), dt.datetime(2026, 1, 1)),
            "get_data": get_solar_orbiter_data,
        },
    }

    # With the positions of the spacecraft we have, we can caluclate the
    # heliocentric distance.
    with spice_client.KernelPool():
        for _, info in spacecraft_info.items():

            times = [
                info["Time Range"][0] + i * TRAJECTORY_RESOLUTION
                for i in range(
                    int(
                        (info["Time Range"][1] - info["Time Range"][0])
                        / TRAJECTORY_RESOLUTION
                    )
                )
            ]
            ets = spice.datetime2et(times)

            positions = spice.spkpos(info["ID"], ets, "J2000", "NONE", "SUN")[0] * u.km

            skycoords = SkyCoord(
                CartesianRepresentation(positions.T),
                obstime=times,
                frame="icrs",
            ).transform_to(frames.HeliographicStonyhurst)

            assert isinstance(skycoords.radius, Quantity)
            assert isinstance(skycoords.lon, Quantity)
            assert isinstance(skycoords.lat, Quantity)

            positions_table = pl.DataFrame(
                {
                    "UTC": times,
                    "Radius [au]": skycoords.radius.to(u.au),
                    "Longitude [deg]": skycoords.lon.to(u.deg),
                    "Latitude [deg]": skycoords.lat.to(u.deg),
                }
            )

            # Filter only to times within AU bounds
            positions_table = positions_table.filter(
                (pl.col("Radius [au]") >= HELIOCENTRIC_DISTANCE_BOUNDS[0])
                & (pl.col("Radius [au]") <= HELIOCENTRIC_DISTANCE_BOUNDS[1])
            )

            # Next we download data for the times where we have positions
            # First lets find all the jumps in time greater than the time resolution
            positions_table = positions_table.with_columns(
                (pl.col("UTC") - pl.col("UTC").shift(1)).alias("Time Step")
            )

            jump_indices = np.where(
                positions_table["Time Step"] > TRAJECTORY_RESOLUTION
            )[0]

            start_indices = np.concatenate(([0], jump_indices)).tolist()
            end_indices = np.concatenate(
                (jump_indices - 1, [len(positions_table) - 1])
            ).tolist()

            data_intervals = [
                TimeRange(positions_table["UTC"][s], positions_table["UTC"][e])
                for s, e in zip(start_indices, end_indices)
            ]

            spacecraft_data = pl.DataFrame()
            for interval in data_intervals:

                # Get data for each interval
                data = info["get_data"](interval)

                spacecraft_data = pl.concat((spacecraft_data, data))

            print(spacecraft_data)


if __name__ == "__main__":
    main()
