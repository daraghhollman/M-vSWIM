"""
Data availability is not the same as spacecraft residence. Just because a
spacecraft is within the orbital range of Mercury, does not mean that the
instruments are on.
"""

import datetime as dt
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import spiceypy as spice
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.units import Quantity
from hermpy.net import ClientSPICE
from matplotlib.axes import Axes
from sunpy.coordinates import frames
from sunpy.time import TimeRange

from mvswim.data import get_helios_data, get_parker_data, get_solar_orbiter_data

# Define the critera with which we look within
HELIOCENTRIC_DISTANCE_BOUNDS = (
    0.3075,
    0.4667,
)  # au (Mercury's perihelion and aphelion)
LATITUDE_BOUND = 3.38  # deg (Mercury's inclination to the Sun's equator)
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
            "Product Name": "psp-fld-l2-mag-rtn-1min",
            "get_data": get_parker_data,
            "Bin Size [sec]": 60,
        },
        "Solar Orbiter": {
            "ID": "Solar Orbiter",
            "Time Range": (dt.datetime(2020, 2, 11), dt.datetime(2026, 1, 1)),
            "Product Name": "mag-rtn-normal-1-minute",
            "get_data": get_solar_orbiter_data,
            "Bin Size [sec]": 60,
        },
        "Helios 1": {
            "ID": "Helios 1",
            "Time Range": (dt.datetime(1974, 12, 11), dt.datetime(1985, 9, 5)),
            "Product Name": "helios1_40sec_mag_plasma",
            "get_data": get_helios1_data,
            "Bin Size [sec]": 40,
        },
        "Helios 2": {
            "ID": "Helios 2",
            "Time Range": (dt.datetime(1976, 1, 16), dt.datetime(1980, 3, 9)),
            "Product Name": "helios2_40sec_mag_plasma",
            "get_data": get_helios2_data,
            "Bin Size [sec]": 40,
        },
    }

    # We cache previous data for quick plotting itteration.
    figure_data_paths = [
        DATA_DIRECTORY / f"cache/data-availability-{s.lower().replace(' ', '_')}.pkl"
        for s in spacecraft_info
    ]

    # Create the directory if it doesn't exist
    os.makedirs(DATA_DIRECTORY / "cache/", exist_ok=True)

    # If the cache files don't exist, create them
    if not all(p.exists() for p in figure_data_paths):

        # With the positions of the spacecraft we have, we can caluclate the
        # heliocentric distance.
        with spice_client.KernelPool():
            for i, (_, info) in enumerate(spacecraft_info.items()):

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

                positions = (
                    spice.spkpos(info["ID"], ets, "J2000", "NONE", "SUN")[0] * u.km
                )

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

                # Filter by latitude
                positions_table = positions_table.filter(
                    np.abs(pl.col("Latitude [deg]")) <= LATITUDE_BOUND
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

                    # Catch edge case where interval is of duration 0
                    # ( Only happens for Helios 2 )
                    if interval.start == interval.end:
                        continue

                    # Get data for each interval
                    data_segments.append(info["get_data"](interval))

                info["Data Segments"] = data_segments

                # Cache data
                with open(figure_data_paths[i], "wb") as f:
                    pickle.dump(info["Data Segments"], f)

    else:
        for i, key in enumerate(spacecraft_info):
            with open(figure_data_paths[i], "rb") as f:
                spacecraft_info[key]["Data Segments"] = pickle.load(f)

    fig, axes_grid = plt.subplots(
        len(spacecraft_info) + 1,
        2,
        height_ratios=[1] + [2] * len(spacecraft_info),
        figsize=(8.3, 11.7),
    )
    axes: List[Axes] = axes_grid.flatten()

    fig.suptitle(
        f"${HELIOCENTRIC_DISTANCE_BOUNDS[0]}"
        + r"\,\rm{AU} \leq R_H \leq"
        + f"{HELIOCENTRIC_DISTANCE_BOUNDS[1]}"
        + r" \,\rm{AU}$"
        + "\n"
        + r"$|\rm{Lat.}| \leq"
        + f" {LATITUDE_BOUND}"
        + r"^\circ $"
    )

    # First make plots of data availability in time
    helios_ax = axes[0]  # Helios 1, 2
    modern_ax = axes[1]  # PSP, SolO, Bepi, etc.

    helios1_data = pl.concat(spacecraft_info["Helios 1"]["Data Segments"])
    helios2_data = pl.concat(spacecraft_info["Helios 2"]["Data Segments"])

    n_timesteps = 10  # to be considered a gap in 'data availability'
    plot_pars = {"lw": 10, "color": "black", "solid_capstyle": "butt"}

    times, y = insert_time_gaps(
        helios1_data,
        time_col="UTC",
        threshold=np.timedelta64(40 * n_timesteps, "s"),  # example threshold
    )
    helios_ax.plot(times, y + 1, **plot_pars)

    times, y = insert_time_gaps(
        helios2_data,
        time_col="UTC",
        threshold=np.timedelta64(40 * n_timesteps, "s"),  # example threshold
    )
    helios_ax.plot(times, y, **plot_pars)

    parker_data = pl.concat(spacecraft_info["Parker Solar Probe"]["Data Segments"])
    solo_data = pl.concat(spacecraft_info["Solar Orbiter"]["Data Segments"])

    times, y = insert_time_gaps(
        parker_data,
        time_col="UTC",
        threshold=np.timedelta64(60 * n_timesteps, "s"),  # example threshold
    )
    modern_ax.plot(times, y + 1, **plot_pars)

    times, y = insert_time_gaps(
        solo_data,
        time_col="UTC",
        threshold=np.timedelta64(60 * n_timesteps, "s"),  # example threshold
    )
    modern_ax.plot(times, y, **plot_pars)

    helios_ax.set_yticks([0, 1, 2, 3], ["", "Helios 2", "Helios 1", ""])
    modern_ax.set_yticks([0, 1, 2, 3], ["", "SolO", "PSP", ""])
    helios_ax.xaxis.set_tick_params(which="major", rotation=60)
    modern_ax.xaxis.set_tick_params(which="major", rotation=60)
    helios_ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    modern_ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    modern_ax.yaxis.set_label_position("right")
    modern_ax.yaxis.tick_right()

    # Then make histograms describing those data
    for i, (name, info) in enumerate(spacecraft_info.items()):

        time_ax = axes[2 * i + 2]
        mag_ax = axes[2 * i + 3]

        # Looping through each data segment, we want to find the following:
        #   - Time differences
        time_differences = pl.Series(dtype=pl.Float64)
        b_total = pl.Series(dtype=pl.Float32)
        br = pl.Series(dtype=pl.Float32)
        bt = pl.Series(dtype=pl.Float32)
        bn = pl.Series(dtype=pl.Float32)

        data_segment: pl.DataFrame
        for data_segment in info["Data Segments"]:

            # We want to know what gaps already exist in this dataset, so we check
            # the distribution of time between each data point.
            data_segment = data_segment.with_columns(
                (pl.col("UTC") - pl.col("UTC").shift(1)).alias("Time Step")
            )

            time_differences.extend(
                data_segment.drop_nulls()
                .get_column("Time Step")
                .dt.total_minutes(fractional=True)
            )

            b_total.extend(data_segment["|B| [nT]"])
            br.extend(data_segment["Br [nT]"])
            bt.extend(data_segment["Bt [nT]"])
            bn.extend(data_segment["Bn [nT]"])

        time_bin_size = info["Bin Size [sec]"] / 60
        time_bins = np.arange(0, max(time_differences) + time_bin_size, time_bin_size)
        time_ax.hist(time_differences.to_list(), color="black", bins=time_bins.tolist())

        time_ax.set_xlim(1, 60)
        # time_ax.set_ylim(0.8, 50)
        time_ax.set_yscale("log")

        time_ax.set_ylabel(
            f"{name}\n" + f"({info["Product Name"]})" + "\n\nN Measurements"
        )

        mag_components = [b_total, br, bt, bn]

        mag_bin_size = 5  # nT
        mag_bins = np.arange(-100, 100 + mag_bin_size, mag_bin_size)

        mag_labels = ["$|B|$", "$B_r$", "$B_t$", "$B_n$"]
        mag_colours = ["black", "#D55E00", "#009E73", "#0072B2"]
        for component, label, colour in zip(mag_components, mag_labels, mag_colours):

            hist_pars = {
                "histtype": "step",
                "color": colour,
                "lw": 3,
                "label": label,
                "bins": mag_bins,
                "density": True,
            }

            mag_ax.hist(component, **hist_pars)

        mag_ax.set_ylabel("Normalised Occurence")
        mag_ax.legend()
        mag_ax.margins(x=0)
        mag_ax.yaxis.set_label_position("right")
        mag_ax.yaxis.tick_right()

    axes[-2].set_xlabel("$\Delta t$ [Minutes]")
    axes[-1].set_xlabel("[nT]")

    fig.subplots_adjust(top=0.9, bottom=0.05, hspace=0.2, wspace=0.1)

    plt.savefig(FIGURE_DIRECTORY / "data-availability.pdf", format="pdf")


def insert_time_gaps(
    df: pl.DataFrame, time_col: str, threshold
) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert NaN gaps where time jumps exceed a threshold.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe
    time_col : str
        Name of datetime column
    threshold : datetime.timedelta
        Maximum allowed time jump before inserting a gap

    Returns
    -------
    times : np.ndarray
        Time array with NaT inserted at gaps
    y : np.ndarray
        Y array (1s) with NaNs inserted at gaps
    """

    df = df.sort(time_col)

    times = df[time_col].to_numpy()
    y = np.ones(len(times))

    diffs = np.diff(times)
    gap_idx = np.where(diffs > threshold)[0] + 1

    times_out = []
    y_out = []

    start = 0
    for g in gap_idx:
        times_out.extend(times[start:g])
        y_out.extend(y[start:g])

        times_out.append(np.datetime64("NaT"))
        y_out.append(np.nan)

        start = g

    times_out.extend(times[start:])
    y_out.extend(y[start:])

    return np.array(times_out), np.array(y_out)


if __name__ == "__main__":
    main()
