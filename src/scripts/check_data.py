"""
When working with the data, I've found that some of our automatically selected
data chunks are returning nan. We need to config this, see if it can be fixed,
and if not, implement some kind of catch for this when we run the model.
"""

import datetime as dt
from typing import Any, Dict, List, Tuple

import astropy.units as u
import numpy as np
import polars as pl
import spiceypy as spice
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.units import Quantity
from hermpy.net import ClientSPICE
from sunpy.coordinates import frames
from sunpy.time import TimeRange

from mvswim.data import get_helios_data, get_parker_data, get_solar_orbiter_data


def main():

    # Simulate state from a config file. See src/scripts/config/
    state = {
        "Config": {
            "data": {
                "spacecraft": [
                    "Solar Orbiter",
                    "Parker Solar Probe",
                    "Helios 1",
                    "Helios 2",
                ],
            },
        }
    }

    # First fetch the required data
    data_chunks, chunk_labels = fetch_data(state)

    label: str
    all_nan_indices = []
    zero_length_indices = []
    for i, (chunk, label) in enumerate(zip(data_chunks, chunk_labels)):

        print(f"Chunk {i+1}/{len(data_chunks)}: {label}")

        # print(chunk)
        # input("")

        if len(chunk) == 0:
            zero_length_indices.append(i)

        if np.isnan(chunk["|B| [nT]"]).all():
            all_nan_indices.append(i)

    print(f"Are there any chunks with all nan data?: {len(all_nan_indices)}")
    print(all_nan_indices)
    print(f"Are there any chunks with length 0?: {len(zero_length_indices)}")
    print(zero_length_indices)


def fetch_data(state: Dict[str, Any]) -> Tuple[List[pl.DataFrame], List[str]]:

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
            "PSP": {
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
    )

    def get_helios1_data(*args):
        return get_helios_data(*args, spacecraft=1)

    def get_helios2_data(*args):
        return get_helios_data(*args, spacecraft=2)

    spacecraft_info: Dict[str, Dict[str, Any]] = {
        "Parker Solar Probe": {
            "ID": "Parker Solar Probe",
            "Time Range": (dt.datetime(2018, 8, 13), dt.datetime(2025, 11, 1)),
            "Product Name": "mag-rtn-normal-1-minute",
            "get_data": get_parker_data,
        },
        "Solar Orbiter": {
            "ID": "Solar Orbiter",
            "Time Range": (dt.datetime(2020, 2, 11), dt.datetime(2026, 1, 1)),
            "Product Name": "psp-fld-l2-mag-rtn-1min",
            "get_data": get_solar_orbiter_data,
        },
        "Helios 1": {
            "ID": "Helios 1",
            "Time Range": (dt.datetime(1974, 12, 11), dt.datetime(1985, 9, 5)),
            "Product Name": "helios1_40sec_mag_plasma",
            "get_data": get_helios1_data,
        },
        "Helios 2": {
            "ID": "Helios 2",
            "Time Range": (dt.datetime(1976, 1, 16), dt.datetime(1980, 3, 9)),
            "Product Name": "helios2_40sec_mag_plasma",
            "get_data": get_helios2_data,
        },
    }

    TRAJECTORY_RESOLUTION = dt.timedelta(hours=1)
    HELIOCENTRIC_DISTANCE_BOUNDS = (0.3, 0.5)  # au

    # Loop through each spacecraft listed in the config file
    data_chunks: List[pl.DataFrame] = []
    data_labels: List[str] = []
    for id in state["Config"]["data"]["spacecraft"]:

        with spice_client.KernelPool():
            info = spacecraft_info[id]
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

            data_segments: List[pl.DataFrame] = []
            for interval in data_intervals:

                # Get data for each interval
                data_segments.append(info["get_data"](interval))

            data_chunks.extend(data_segments)
            data_labels.extend([id] * len(data_segments))

    return data_chunks, data_labels


if __name__ == "__main__":
    main()
