import datetime as dt
import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import requests

DATA_DIRECTORY = Path(__file__).parent.parent.parent / "data/"
FIGURE_DIRECTORY = Path(__file__).parent.parent.parent / "figures/"


def main() -> None:
    sunspots = get_sunspot_number()

    fig, ax = plt.subplots(figsize=(8.3, 3))

    plot_start_year = 1974
    plot_end_year = 2026

    # Filter sunspots to only those in the range of our spacecraft.
    sunspots = sunspots.filter(
        pl.col("Decimal Year").is_between(plot_start_year, plot_end_year)
    )

    ax.plot(
        sunspots["Decimal Year"],
        sunspots["Mean"],
        color="cornflowerblue",
        label="Mean",
    )

    # 95% confidence interval
    upper = sunspots["Mean"] + 1.96 * sunspots["Standard Deviation"]
    lower = sunspots["Mean"] - 1.96 * sunspots["Standard Deviation"]

    ax.fill_between(
        sunspots["Decimal Year"],
        lower,
        upper,
        color="cornflowerblue",
        alpha=0.2,
        label="95% CI",
    )

    # Add each spacecraft
    spacecraft_names = ["SolO", "PSP", "BepiColombo", "H1", "H2"]
    spacecraft_starts = [
        dt.datetime(2020, 2, 11),
        dt.datetime(2018, 8, 13),
        dt.datetime(2018, 10, 21),
        dt.datetime(1975, 1, 16),
        dt.datetime(1976, 7, 21),
    ]

    spacecraft_ends = [
        dt.datetime.today(),
        dt.datetime.today(),
        dt.datetime.today(),
        dt.datetime(1985, 2, 18),
        dt.datetime(1979, 12, 23),
    ]

    spacecraft_starts = [datetime_to_decimal_year(t) for t in spacecraft_starts]
    spacecraft_ends = [datetime_to_decimal_year(t) for t in spacecraft_ends]

    heights = [200, 100, 50, 150, 200]
    for start, end, label, height in zip(
        spacecraft_starts, spacecraft_ends, spacecraft_names, heights
    ):
        range_indicator(
            ax,
            start,
            end,
            height,
            label,
            cap_height=10,
            label_offset=10,
            color="black" if label != "BepiColombo" else "dimgrey",
        )

    ax.set_ylabel("Monthly Sunspot Number")
    ax.margins(x=0)
    ax.legend()

    plt.savefig(FIGURE_DIRECTORY / "spacecraft-overview.pdf", format="pdf")


def get_sunspot_number() -> pl.DataFrame:
    """Fetches a timeseries of sunspot number"""

    url = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php"
    path = DATA_DIRECTORY / "monthly_sunspot_number.csv"

    # Check if the file exists before downloading:
    if not os.path.exists(path):
        response = requests.get(url)
        with open(path, "wb") as file:
            file.write(response.content)

    data = pl.read_csv(
        path,
        new_columns=[
            "Year",
            "Month",
            "Decimal Year",
            "Mean",
            "Standard Deviation",
            "N Observations",
            "Provisional Marker",
        ],
        schema_overrides={
            "Mean": pl.Float64,
            "Standard Deviation": pl.Float64,
            "N Observations": pl.Float64,
        },
        has_header=False,
        separator=";",
    )

    return data


def datetime_to_decimal_year(d: dt.datetime) -> float:
    year_start = dt.datetime(d.year, 1, 1)
    year_end = dt.datetime(d.year + 1, 1, 1)
    year_length = (year_end - year_start).total_seconds()

    elapsed = (d - year_start).total_seconds()

    return d.year + elapsed / year_length


def range_indicator(
    ax,
    x_start,
    x_end,
    y_level,
    label="",
    cap_height=0.05,
    color="black",
    linewidth=1.5,
    label_offset=0.1,
    fontsize=10,
):
    ax.hlines(y_level, x_start, x_end, colors=color, linewidth=linewidth)
    ax.vlines(
        x_start,
        y_level - cap_height,
        y_level + cap_height,
        colors=color,
        linewidth=linewidth,
    )
    ax.vlines(
        x_end,
        y_level - cap_height,
        y_level + cap_height,
        colors=color,
        linewidth=linewidth,
    )

    if label:
        ax.text(
            (x_start + x_end) / 2,
            y_level - label_offset,
            label,
            ha="center",
            va="top",
            fontsize=fontsize,
            color=color,
        )


if __name__ == "__main__":
    main()
