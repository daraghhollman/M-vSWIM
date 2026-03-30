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
import functools
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import astropy.units as u
import numpy as np
import polars as pl
import pynvml
import spiceypy as spice
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.units import Quantity
from hermpy.net import ClientSPICE
from numpy.typing import NDArray
from sunpy.coordinates import frames
from sunpy.time import TimeRange

from mvswim.data import get_helios_data, get_parker_data, get_solar_orbiter_data
from mvswim.modelling import SolarWindModel
from mvswim.scalling import TimeScaler

LOG_DIR = Path(__file__).parent.parent.parent / "logs"


def main():

    state: Dict[str, Any] = {}

    # Check for gpu
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"Found {device_count} GPU devices")

        state["GPU"] = True

    except pynvml.NVMLError as e:
        print(f"No NVIDIA GPU detected: {e}")

        state["GPU"] = False

    # Start by checking that the config is correct, and
    # parsing the needed elements.
    state["Config"] = parse_config(state)

    if state["Config"] is None:
        return

    # First fetch the required data
    data_chunks, chunk_labels = fetch_data(state)

    # Apply model to each chunk
    metrics: None | pl.DataFrame = None
    chunk: pl.DataFrame
    label: str
    for i, (chunk, label) in enumerate(zip(data_chunks, chunk_labels)):

        log(f"Applying model to chunk: {i+1}/{len(data_chunks)}", state)
        chunk_metrics = apply_model(chunk, state)
        chunk_metrics.update({"Spacecraft": label})

        # Flatten our nested metrics
        flat_metrics = {}
        for key, value in chunk_metrics.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    flat_metrics[f"{key} {subkey}"] = float(subval)
            else:
                flat_metrics[key] = value

        if metrics is None:
            metrics = pl.DataFrame(flat_metrics)

        else:
            metrics.extend(pl.DataFrame(flat_metrics))

    assert isinstance(metrics, pl.DataFrame)

    # Save metrics
    metrics.write_csv(state["Log Directory"] / "performance-metrics.csv")

    # Cleanup
    if state["GPU"] is True:
        pynvml.nvmlShutdown()


def apply_model(data_chunk: pl.DataFrame, state: Dict[str, Any]) -> Dict[str, Any]:

    # For now we will start with just using |B|.
    X: NDArray = data_chunk["UTC"].to_numpy().reshape(-1, 1)
    Y: NDArray = data_chunk["|B| [nT]"].to_numpy().reshape(-1, 1).astype("float64")

    # Remove any nans
    valid_mask = ~np.isnan(Y).squeeze()
    X = X[valid_mask]
    Y = Y[valid_mask]

    time_scaler = TimeScaler(X)

    # First we need to create our artificial gaps
    gap_generator = state["Config"]["Model"]["Gap Generator"]

    training_x, training_y, testing_x, testing_y = gap_generator.create_gaps(X, Y)

    # Then we're gonna apply the model, quantify performance, and produce a figure.
    model = SolarWindModel.build(
        input=training_x,
        output=training_y,
        time_scaler=time_scaler,
        n_inducing_points=state["Config"]["Model"]["Inducing Points"],
        log_directory=state["Log Directory"],
        seed=state["Config"]["Seed"],
    )

    model.train_model(log_gpu=state["GPU"])
    performance_metrics = model.test_performance(testing_x, testing_y)
    model.quicklook(testing_data=(testing_x, testing_y))

    return performance_metrics


def parse_config(state: Dict[str, Any]) -> Dict[str, Any] | None:

    # First check the format of the input, we expect the following call:
    # python ./src/scripts/run_model.py -c <config.toml>
    if len(sys.argv) != 2:
        print("Incorrect syntax. A config file must be provided.")
        print("")
        print("Usage:")
        print("python ./src/scripts/run_model.py <config.py>")
        print("")
        print("See ./src/scripts/config/ for examples.")
        return None

    # Import the config dict from file
    spec = importlib.util.spec_from_file_location("config", sys.argv[1])

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    state["Config"] = module.CONFIG

    # Create a logfile corresponding to the filename of the config file and the
    # current time.
    try:
        state_log_dir = (
            state["Config"]["Log Directory"]
            / f"{dt.datetime.now().strftime('%Y-%m-%d--%H:%M:%S')}--{Path(sys.argv[1]).name.split('.')[0]}"
        )

    except KeyError:
        state_log_dir = (
            LOG_DIR
            / f"{dt.datetime.now().strftime('%Y-%m-%d--%H:%M:%S')}--{Path(sys.argv[1]).name.split('.')[0]}"
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
    for id in state["Config"]["Data"]["Spacecraft"]:
        if id not in spacecraft_options:
            log(f"Config Error: Invalid input in data.spacecraft: {id}", state)
            log(f"Available options are: {spacecraft_options}", state)
            log("Exitting.", state)
            return None

    log("Successfully parse config file", state)
    return state["Config"]


def fetch_data(state: Dict[str, Any]) -> Tuple[List[pl.DataFrame], List[str]]:

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

    def get_helios1_data(*args, **kwargs):
        return get_helios_data(*args, **kwargs, spacecraft=1)

    def get_helios2_data(*args, **kwargs):
        return get_helios_data(*args, **kwargs, spacecraft=2)

    spacecraft_info: Dict[str, Dict[str, Any]] = {
        "Parker Solar Probe": {
            "ID": "Parker Solar Probe",
            "Time Range": (dt.datetime(2018, 8, 13), dt.datetime(2025, 11, 1)),
            "get_data": functools.partial(
                get_parker_data, product="psp-fld-l2-mag-rtn-1min"
            ),
        },
        "Solar Orbiter": {
            "ID": "Solar Orbiter",
            "Time Range": (dt.datetime(2020, 2, 11), dt.datetime(2026, 1, 1)),
            "get_data": functools.partial(
                get_solar_orbiter_data,
                product="mag-rtn-normal-1-minute",
                quality_limit=2,
            ),
        },
        "Helios 1": {
            "ID": "Helios 1",
            "Time Range": (dt.datetime(1974, 12, 11), dt.datetime(1985, 9, 5)),
            "Product Name": "40sec_mag_plasma",
            "get_data": functools.partial(get_helios1_data, product="40sec_mag_plasma"),
        },
        "Helios 2": {
            "ID": "Helios 2",
            "Time Range": (dt.datetime(1976, 1, 16), dt.datetime(1980, 3, 9)),
            "get_data": functools.partial(get_helios2_data, product="40sec_mag_plasma"),
        },
    }

    filter_trajectory_resolution = state["Config"]["Data"]["Filter"][
        "Filter Resolution"
    ]
    heliocentric_distance_bounds = state["Config"]["Data"]["Filter"][
        "Heliocentric Distance [AU]"
    ]
    latitude_limit = state["Config"]["Data"]["Filter"]["Latitude [deg]"]

    # Loop through each spacecraft listed in the config file
    log(f"Filter parameters: {heliocentric_distance_bounds} au", state)
    log(f"Filter parameters: {latitude_limit} deg", state)
    log("Fetching filtered spacecraft data for:", state)
    data_chunks: List[pl.DataFrame] = []
    data_labels: List[str] = []
    for id in state["Config"]["Data"]["Spacecraft"]:

        log(f"   {id}", state)

        with spice_client.KernelPool():
            info = spacecraft_info[id]
            times = [
                info["Time Range"][0] + i * filter_trajectory_resolution
                for i in range(
                    int(
                        (info["Time Range"][1] - info["Time Range"][0])
                        / filter_trajectory_resolution
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
                (pl.col("Radius [au]") >= heliocentric_distance_bounds[0])
                & (pl.col("Radius [au]") <= heliocentric_distance_bounds[1])
            )

            # Next we download data for the times where we have positions
            # First lets find all the jumps in time greater than the time resolution
            positions_table = positions_table.with_columns(
                (pl.col("UTC") - pl.col("UTC").shift(1)).alias("Time Step")
            )

            jump_indices = np.where(
                positions_table["Time Step"] > filter_trajectory_resolution
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

    log("Data fetch complete", state)

    return data_chunks, data_labels


def log(s: str, state: Dict[str, Any]) -> None:

    # Print the statement
    print(s)

    # Log the statement to a file
    with open(state["Log File"], "a") as f:
        f.write(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S  ") + s + "\n")


if __name__ == "__main__":
    main()
