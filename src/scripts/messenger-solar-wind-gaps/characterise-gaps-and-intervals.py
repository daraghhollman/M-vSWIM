"""
We need to model the distribution of MESSENGER's solar wind gaps and intervals
to be able to sample them on the fly.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
import scipy.stats

DATA_DIRECTORY = Path(__file__).parent.parent.parent.parent / "data/"


def main() -> None:

    intervals = get_solar_wind_intervals(get_crossing_list())
    gaps = get_solar_wind_gaps(intervals)

    # When modelling gaps, we chose to discard durations less than one hour, as
    # these are systematically different to the longer gaps which we are
    # interested in capturing.
    interval_durations = intervals.filter(pl.col("Duration") > pl.duration(hours=1))[
        "Duration"
    ].dt.total_hours(fractional=True)

    gap_durations = gaps.filter(pl.col("Duration") > pl.duration(hours=1))[
        "Duration"
    ].dt.total_hours(fractional=True)

    # Make a kde estimate
    intervals_kde = scipy.stats.gaussian_kde(interval_durations)
    gaps_kde = scipy.stats.gaussian_kde(gap_durations)

    bins = np.arange(0, 20, 0.1)
    bin_centres = (bins[:-1] + bins[1:]) / 2

    intervals_kde_evaluation = intervals_kde.evaluate(bin_centres)
    gaps_kde_evaluation = gaps_kde.evaluate(bin_centres)

    _, axes = plt.subplots(4, 1, height_ratios=[1, 4, 1, 4], sharex=True)
    interval_kde_ax, interval_ax, gap_kde_ax, gap_ax = axes

    # INTERVALS
    # Plot full interval dataset
    interval_ax.hist(
        intervals["Duration"].dt.total_hours(fractional=True),
        color="lightgrey",
        label="Full Intervals Distribution",
        bins=bins.tolist(),
    )

    # Plot the distribution of gaps we are interested in.
    interval_ax.hist(
        interval_durations,
        color="black",
        label="Intervals of Interest",
        bins=bins.tolist(),
    )

    interval_ax.set_xlabel("Duration [hours]")
    interval_ax.set_ylabel("Number of Intervals")
    interval_ax.set_ylim(0, 1000)
    interval_ax.legend()

    # Plot the kde evaluation
    interval_kde_ax.plot(bin_centres, intervals_kde_evaluation, color="black")

    interval_kde_ax.set_ylabel("Kernel Density Estimate")
    interval_kde_ax.set_ylim(0, 0.6)

    # GAPS
    # Plot full gap dataset
    gap_ax.hist(
        gaps["Duration"].dt.total_hours(fractional=True),
        color="lightgrey",
        label="Full Gaps Distribution",
        bins=bins.tolist(),
    )

    # Plot the distribution of gaps we are interested in.
    gap_ax.hist(
        gap_durations,
        color="black",
        label="Gaps of Interest",
        bins=bins.tolist(),
    )

    gap_ax.set_xlabel("Duration [hours]")
    gap_ax.set_ylabel("Number of Gaps")
    gap_ax.set_xlim(0, 12)
    gap_ax.set_ylim(0, 1000)
    gap_ax.legend()

    # Plot the kde evaluation
    gap_kde_ax.plot(bin_centres, gaps_kde_evaluation, color="black")

    gap_kde_ax.set_ylabel("Kernel Density Estimate")
    gap_kde_ax.set_ylim(0, 0.6)

    plt.tight_layout()
    plt.show()


def get_crossing_list() -> pl.DataFrame:

    # Download the Hollman et al. (2026) crossing list
    url = "https://zenodo.org/records/17814795/files/hollman_2025_crossing_list.csv?download=1"
    crossing_list_path = DATA_DIRECTORY / "hollman_2026_crossing_list.csv"

    # If the file doesn't exist, download it
    if not os.path.exists(crossing_list_path):
        response = requests.get(url)
        with open(crossing_list_path, "wb") as file:
            file.write(response.content)

    return pl.read_csv(crossing_list_path, try_parse_dates=True).rename({"Time": "UTC"})


def get_solar_wind_intervals(crossings: pl.DataFrame) -> pl.DataFrame:

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

    # Add their duration
    intervals = intervals.with_columns(
        (pl.col("End Time") - pl.col("Start Time")).alias("Duration"),
    )

    return intervals


def get_solar_wind_gaps(solar_wind_intervals: pl.DataFrame) -> pl.DataFrame:
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

    # Add their durations
    gaps = gaps.with_columns(
        (pl.col("End Time") - pl.col("Start Time")).alias("Duration"),
    )

    return gaps


if __name__ == "__main__":
    main()
