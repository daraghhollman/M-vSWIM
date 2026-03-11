"""
We need a script to make quick and easy iterrations of model runs. Ideally,
this will take some input file which contains config for how to run the model,
and copies it to the log folder for reproducability.

This script will apply MESSENGER-like gaps to a selection of Solar Orbiter,
Parker Solar Probe, BepiColombo, and Helios 1/2. A mdoel will be constructed
and fit to the remaining data, computing metrics of performance on the data
during the artificial gaps. These metrics will be quantified and logged along
with quicklook plots, and other useful outputs.
"""

import datetime as dt
import os
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List

import astropy.units as u
import numpy as np
import polars as pl
import spiceypy as spice
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.units import Quantity
from hermpy.net import ClientSPICE
from numpy.typing import NDArray
from sunpy.coordinates import frames
from sunpy.time import TimeRange

from mvswim.data import get_helios_data, get_parker_data, get_solar_orbiter_data
from mvswim.modelling import GapGenerator, SolarWindModel

LOG_DIR = Path(__file__).parent.parent.parent / "logs"


def main():

    state: Dict[str, Any] = {}

    # Start by checking that the config is correct, and
    # parsing the needed elements.
    state["Config"] = parse_config(state)

    if state["Config"] is None:
        return

    # First fetch the required data
    data_chunks = fetch_data(state)

    # Apply model to each chunk
    chunk: pl.DataFrame
    for chunk in data_chunks:
        apply_model(chunk, state)


def apply_model(data_chunk: pl.DataFrame, state: Dict[str, Any]) -> None:

    # For now we will start with just using |B|.
    X: NDArray = data_chunk["UTC"].to_numpy().reshape(-1, 1)
    Y: NDArray = data_chunk["|B| [nT]"].to_numpy().reshape(-1, 1).astype("float64")

    # First we need to create our artificial gaps
    # Need to consider changing this to use time differences rather than
    # indexing, but for now this works with SolO and PSP as they have regular
    # data.
    # 2 hour gaps, at a 4 hour interval, with standard deviations of 30 minutes.
    gap_generator = GapGenerator.from_normal_distributions(2 * 60, 30, 4 * 60, 30)

    # Task for tomorrow! Make gap generator return both training and testing
    # data, rather than just filtering to training data.
    X, Y = gap_generator.create_gaps(X, Y)

    # Then we're gonna apply the model, quantify performance, and produce a figure.

    # model = SolarWindModel.build(
    #     input=X,
    #     output=Y,
    #     n_inducing_points=50,
    #     log_directory=LOG_DIR,
    #     seed=1785,
    # )
    #
    # model.train_model()
    #
    # model.quicklook()


def parse_config(state: Dict[str, Any]) -> Dict[str, Any] | None:

    # First check the format of the input, we expect the following call:
    # python ./src/scripts/run_model.py -c <config.toml>
    if len(sys.argv) != 2:
        print("Incorrect syntax. A config file must be provided.")
        print("")
        print("Usage:")
        print("python ./src/scripts/run_model.py <config-file.toml>")
        return None

    with open(sys.argv[1], "rb") as f:
        state["Config"] = tomllib.load(f)

    # Create a logfile corresponding to the filename of the config file and the
    # current time.
    state_log_dir = (
        LOG_DIR
        / f"{Path(sys.argv[1]).name.split('.')[0]}--{dt.datetime.now().strftime('%Y-%m-%d--%H:%M:%S')}"
    )
    os.makedirs(state_log_dir, exist_ok=True)
    state["Log Directory"] = state_log_dir
    state["Log File"] = state["Log Directory"] / "log"

    # If we can at least parse the config as a toml file, we should copy it to
    # the log directory. This will ensure every run is entirely reproducable.
    shutil.copy(Path(sys.argv[1]), state["Log Directory"])

    # There must be a data section which defines which spacecraft to work with.
    # There is an exhaustive list of options.
    spacecraft_options = set(
        ["Solar Orbiter", "Parker Solar Probe", "Helios 1", "Helios 2"]
    )
    for id in state["Config"]["data"]["spacecraft"]:
        if id not in spacecraft_options:
            log(f"Config Error: Invalid input in data.spacecraft: {id}", state)
            log(f"Available options are: {spacecraft_options}", state)
            log("Exitting.", state)
            return None

    log("Successfully parse config file", state)
    return state["Config"]


def fetch_data(state: Dict[str, Any]) -> List[pl.DataFrame]:

    log("Setting up spice client", state)
    spice_client = ClientSPICE()

    log("Updating spice kernels", state)
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
    log(f"Filter parameters: {HELIOCENTRIC_DISTANCE_BOUNDS} au", state)
    log("Fetching filtered spacecraft data for:", state)
    data_chunks: List[pl.DataFrame] = []
    for id in state["Config"]["data"]["spacecraft"]:

        log(f"   {id}", state)

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

    log("Data fetch complete", state)

    return data_chunks


def log(s: str, state: Dict[str, Any]) -> None:

    # Print the statement
    print(s)

    # Log the statement to a file
    with open(state["Log File"], "a") as f:
        f.write(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S  ") + s + "\n")


if __name__ == "__main__":
    main()
