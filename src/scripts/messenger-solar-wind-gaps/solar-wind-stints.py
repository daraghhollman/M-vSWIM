"""
We are interested in knowing the lengths of continous time spent by MESSENGER
in the solar wind. We base this on the crossing list by Hollman et al. (2026).

We are also interested in looking at how these change with Mercury's precession
around the sun. We can look at four groups in heliocentric distance: All, <
0.36 AU, between 0.36 AU and 0.41 AU, and > 0.41.
"""

import os
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
import spiceypy as spice
from hermpy.net import ClientSPICE
from matplotlib.ticker import MultipleLocator

from vswim.orbit_numbers import get_orbit_number

DATA_DIRECTORY = Path(__file__).parent.parent.parent.parent / "data/"
FIGURE_DIRECTORY = Path(__file__).parent.parent.parent.parent / "figures/"


def main():

    # Check if file exists before attempting to recreate
    sw_cache_path = DATA_DIRECTORY / "solar_wind_intervals.parquet"
    gap_cache_path = DATA_DIRECTORY / "gap_intervals.parquet"

    if not os.path.exists(sw_cache_path) or not os.path.exists(gap_cache_path):
        crossings = load_crossings()
        solar_wind_intervals = find_solar_wind_intervals(crossings)
        solar_wind_intervals.write_parquet(sw_cache_path)

        solar_wind_gaps = find_solar_wind_gaps(solar_wind_intervals)
        solar_wind_gaps.write_parquet(gap_cache_path)

    else:
        solar_wind_intervals = pl.read_parquet(sw_cache_path)
        solar_wind_gaps = pl.read_parquet(gap_cache_path)

    print("Solar Wind Intervals")
    print(solar_wind_intervals)
    print("Solar Wind Gaps")
    print(solar_wind_gaps)

    # Do some cool plotting

    plot_solar_wind_intervals(solar_wind_intervals)
    plot_solar_wind_gaps(solar_wind_gaps)

    plot_time_per_orbit(solar_wind_intervals)
    scatter_sw_time_vs_heliocentric_distance(solar_wind_intervals)


def scatter_sw_time_vs_heliocentric_distance(
    solar_wind_intervals: pl.DataFrame,
) -> None:

    orbits = (
        solar_wind_intervals.group_by("Orbit Number")
        .agg(pl.col("Duration").sum())
        .sort("Orbit Number")
    )

    distances = (
        solar_wind_intervals.group_by("Orbit Number")
        .agg(pl.col("Heliocentric Distance [au]").mean())
        .sort("Orbit Number")
    )
    distances = distances["Heliocentric Distance [au]"]
    duration = orbits["Duration"].dt.total_hours(fractional=True)

    _, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(
        distances,
        duration,
        color="black",
        marker=".",
    )

    plt.show()


def plot_solar_wind_intervals(solar_wind_intervals: pl.DataFrame) -> None:
    _, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
    axes = axes.flatten()

    # Define some conditions on heliocentric distance
    filters = [
        solar_wind_intervals["Heliocentric Distance [au]"] < 0.36,
        (solar_wind_intervals["Heliocentric Distance [au]"] >= 0.36)
        & (solar_wind_intervals["Heliocentric Distance [au]"] < 0.41),
        solar_wind_intervals["Heliocentric Distance [au]"] >= 0.41,
    ]

    labels = [
        r"All",
        "Perihelion\n" + r"$R_{\rm H}$ < 0.36 AU",
        r"0.36 AU $\leq R_{\rm H}$ < 0.41 AU",
        "Aphelion\n" + r"0.41 AU $\leq R_{\rm H}$",
    ]

    bins = np.arange(0, 12 + 0.5, 0.5)

    for i, ax in enumerate(axes):

        if i == 0:
            # No filter for the first plot
            filtered_intervals = solar_wind_intervals

        else:
            filtered_intervals = solar_wind_intervals.filter(filters[i - 1])

        ax.hist(
            filtered_intervals["Duration"].dt.total_hours(fractional=True),
            bins=bins,
            color="grey",
        )

        inset_ax = ax.inset_axes([0.25, 0.45, 0.7, 0.45])
        inset_ax.hist(
            filtered_intervals["Duration"].dt.total_hours(fractional=True),
            bins=bins[4:],
            color="grey",
        )

        ax.set_title(labels[i])

        ax.margins(x=0)
        inset_ax.margins(x=0)

        ax.xaxis.set_major_locator(MultipleLocator(2))
        inset_ax.xaxis.set_major_locator(MultipleLocator(2))

        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        inset_ax.xaxis.set_minor_locator(MultipleLocator(0.5))

        if i % 2 == 0:
            ax.set_ylabel("Number of Solar Wind Stints")

        if i > 1:
            ax.set_xlabel("Solar Wind Stint Duration [hours]")

    plt.tight_layout()
    plt.savefig(FIGURE_DIRECTORY / "solar-wind-stints.pdf", format="pdf")


def plot_solar_wind_gaps(solar_wind_gaps: pl.DataFrame) -> None:
    _, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
    axes = axes.flatten()

    # Define some conditions on heliocentric distance
    filters = [
        solar_wind_gaps["Heliocentric Distance [au]"] < 0.36,
        (solar_wind_gaps["Heliocentric Distance [au]"] >= 0.36)
        & (solar_wind_gaps["Heliocentric Distance [au]"] < 0.41),
        solar_wind_gaps["Heliocentric Distance [au]"] >= 0.41,
    ]

    labels = [
        r"All",
        "Perihelion\n" + r"$R_{\rm H}$ < 0.36 AU",
        r"0.36 AU $\leq R_{\rm H}$ < 0.41 AU",
        "Aphelion\n" + r"0.41 AU $\leq R_{\rm H}$",
    ]

    bins = np.arange(0, 6 + 0.5, 0.5)

    for i, ax in enumerate(axes):

        if i == 0:
            # No filter for the first plot
            filtered_intervals = solar_wind_gaps

        else:
            filtered_intervals = solar_wind_gaps.filter(filters[i - 1])

        ax.hist(
            filtered_intervals["Duration"].dt.total_hours(fractional=True),
            bins=bins,
            color="grey",
        )

        inset_ax = ax.inset_axes([0.25, 0.45, 0.7, 0.45])
        inset_ax.hist(
            filtered_intervals["Duration"].dt.total_hours(fractional=True),
            bins=bins[2:],
            color="grey",
        )

        ax.set_title(labels[i])

        ax.margins(x=0)
        inset_ax.margins(x=0)

        ax.xaxis.set_major_locator(MultipleLocator(2))
        inset_ax.xaxis.set_major_locator(MultipleLocator(2))

        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        inset_ax.xaxis.set_minor_locator(MultipleLocator(0.5))

        if i % 2 == 0:
            ax.set_ylabel("Number of Solar Wind Gaps")

        if i > 1:
            ax.set_xlabel("Solar Wind Gap Duration [hours]")

    plt.tight_layout()
    plt.savefig(FIGURE_DIRECTORY / "solar-wind-gaps.pdf", format="pdf")


def load_crossings():

    for d in [DATA_DIRECTORY, FIGURE_DIRECTORY]:
        if not os.path.isdir(d):
            os.makedirs(d)

    # Download the Hollman et al. (2026) crossing list
    url = "https://zenodo.org/records/17814795/files/hollman_2025_crossing_list.csv?download=1"
    crossing_list_path = DATA_DIRECTORY / "hollman_2026_crossing_list.csv"

    # If the file doesn't exist, download it
    if not os.path.exists(crossing_list_path):
        response = requests.get(url)
        with open(crossing_list_path, "wb") as file:
            file.write(response.content)

    return pl.read_csv(crossing_list_path, try_parse_dates=True).rename({"Time": "UTC"})


def find_solar_wind_intervals(crossings: pl.DataFrame):

    intervals = (
        crossings.with_columns(
            pl.col("UTC").shift(-1).alias("Next UTC"),
            pl.col("Label").shift(-1).alias("Next Label"),
        )
        # Filter to only BS_OUTs followed by BS_INs
        .filter(
            (pl.col("Label") == "BS_OUT") & (pl.col("Next Label") == "BS_IN")
        ).select(
            [
                pl.col("UTC").alias("Start Time"),
                pl.col("Next UTC").alias("End Time"),
            ]
        )
    )

    # Add their length and middle time.
    intervals = intervals.with_columns(
        (pl.col("End Time") - pl.col("Start Time")).alias("Duration"),
        (pl.col("Start Time") + (pl.col("End Time") - pl.col("Start Time"))).alias(
            "Mid Time"
        ),
    )

    # Add orbit number to each interval
    intervals = intervals.with_columns(
        pl.col("Start Time")
        .map_elements(get_orbit_number, return_dtype=pl.Int64)
        .alias("Orbit Number")
    )

    # Add the heliocentric distance of Mercury at the mid time
    spice_client = ClientSPICE()

    with spice_client.KernelPool():
        intervals = intervals.with_columns(
            pl.col("Mid Time")
            .map_batches(
                get_heliocentric_distances,
                return_dtype=pl.Float64,
            )
            .alias("Heliocentric Distance [au]")
        )

    # Remove the mid time from the dataframe
    intervals = intervals.drop("Mid Time")

    return intervals


def find_solar_wind_gaps(solar_wind_intervals: pl.DataFrame):
    # The gaps in the solar wind will simply be the spans of time between each
    # interval of solar wind.

    gaps = (
        solar_wind_intervals.with_columns(
            pl.col("End Time").shift().alias("Previous End")
        )
        .with_columns(
            # Make a column of "Gap"s which contains essentially its own row
            # within based on that row's data.
            pl.when(pl.col("Start Time") > pl.col("Previous End"))
            .then(
                pl.struct(
                    pl.col("Previous End").alias("Start Time"),
                    pl.col("Start Time").alias("End Time"),
                )
            )
            .otherwise(None)
            .alias("Gap")
        )
        .select("Gap")
        .drop_nulls()
        .unnest("Gap")
    )

    # Add orbit number to each interval
    gaps = gaps.with_columns(
        pl.col("Start Time")
        .map_elements(get_orbit_number, return_dtype=pl.Int64)
        .alias("Orbit Number")
    )

    # Add their length and middle time.
    gaps = gaps.with_columns(
        (pl.col("End Time") - pl.col("Start Time")).alias("Duration"),
        (pl.col("Start Time") + (pl.col("End Time") - pl.col("Start Time"))).alias(
            "Mid Time"
        ),
    )

    # Add the heliocentric distance of Mercury at the mid time
    spice_client = ClientSPICE()

    with spice_client.KernelPool():
        gaps = gaps.with_columns(
            pl.col("Mid Time")
            .map_batches(
                get_heliocentric_distances,
                return_dtype=pl.Float64,
            )
            .alias("Heliocentric Distance [au]")
        )

    # Remove the mid time from the dataframe
    gaps = gaps.drop("Mid Time")

    return gaps


def get_heliocentric_distances(times: pl.Series) -> pl.Series:
    dt_times = times.to_list()

    ets = spice.datetime2et(dt_times)
    positions, _ = spice.spkpos("MERCURY", ets, "J2000", "NONE", "SUN")

    distances = np.sqrt(np.sum(positions**2, axis=1))

    # Convert from km to AU
    distances *= u.km
    distances = distances.to_value("au")

    return pl.Series(distances)


def plot_time_per_orbit(intervals: pl.DataFrame) -> None:

    # We first need to sum all the durations for the same orbit number
    orbits = (
        intervals.group_by("Orbit Number")
        .agg(pl.col("Duration").sum())
        .sort("Orbit Number")
    )

    distances = (
        intervals.group_by("Orbit Number")
        .agg(pl.col("Heliocentric Distance [au]").mean())
        .sort("Orbit Number")
    )

    _, ax = plt.subplots(figsize=(6, 4))

    duration = orbits["Duration"].dt.total_hours(fractional=True)
    ax.plot(
        orbits["Orbit Number"],
        duration,
        color="grey",
        label="Total Solar Wind Duration",
    )

    # A smoothed average
    m = 20
    ax.plot(
        orbits["Orbit Number"],
        np.convolve(duration, np.ones(m) / m, mode="same"),
        color="black",
        label=f"Moving Average over $M = {m}$ orbits",
    )

    # Lets also plot heliocentric distance for comparison
    twin = ax.twinx()
    twin.plot(
        distances["Orbit Number"],
        distances["Heliocentric Distance [au]"],
        color="cornflowerblue",
        alpha=0.5,
    )

    twin.yaxis.set_ticks(
        np.arange(0, 0.5 + 0.05, 0.05),
        # labels=[None] * 6 + ["0.30", "0.35", "0.40", "0.45", "0.50"],
    )

    twin.yaxis.label.set_color("cornflowerblue")
    twin.tick_params(axis="y", colors="cornflowerblue")
    twin.spines["right"].set_edgecolor("cornflowerblue")

    twin.set_ylabel("Heliocentric Distance [au]")

    ax.set_xlabel("Orbit Number")
    ax.set_ylabel("Time in Solar Wind [hours]")

    ax.legend(loc="center right")

    plt.tight_layout()
    plt.savefig(FIGURE_DIRECTORY / "solar-wind-per-orbit.pdf", format="pdf")


if __name__ == "__main__":
    main()
