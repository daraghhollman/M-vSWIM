"""
Script to plot data from spacecraft in the solar wind and extract metrics in a
table below.
"""

import datetime as dt
from typing import Callable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import spiceypy as spice
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, SkyCoord
from hermpy.net import ClientSPICE
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter
from sunpy.coordinates import frames
from sunpy.time import TimeRange

from mvswim.data import get_solar_orbiter_data


def main() -> None:

    time_range = TimeRange("2020-2-11", "2026-01-01")

    data: pl.DataFrame = get_solar_orbiter_data(
        time_range,
        "mag-rtn-normal-1-minute",
        quality_limit=2,
    ).drop_nans()

    # Near mercury data
    near_mercury_data: List[pl.DataFrame] = get_times_near_mercury(time_range, data)

    # Get ICME list
    url = "https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v23.csv"
    icmecat = (
        pl.read_csv(url, try_parse_dates=True)
        .rename(
            {
                "icmecat_id": "ID",
                "sc_insitu": "Spacecraft",
                "icme_start_time": "Start Time",
                "icme_duration": "Duration",
            }
        )
        .filter(pl.col("Spacecraft") == "Solar Orbiter")
    )

    # Loop through near-Mercury data chunks
    for i, data_chunk in enumerate(near_mercury_data):

        print(f"Displaying chunk {i+1}/{len(near_mercury_data)}", end="\r")

        _, ax = plt.subplots()

        # MAG Axis
        create_plot(ax, data_chunk)

        ax.set(
            ylabel="nT",
            ylim=(-max(data_chunk["|B| [nT]"]), max(data_chunk["|B| [nT]"])),
        )

        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

        ax.legend()
        ax.margins(x=0)

        # Table
        row_labels = ["Mean", "Median", "Standard Deviation"]
        column_labels = [
            "|B| [nT]",
            "Br [nT]",
            "Bt [nT]",
            "Bn [nT]",
        ]  # All but time column
        metrics: List[Callable] = [np.mean, np.median, np.std]

        cell_text: List[List[str]] = []
        for func in metrics:
            row_text: List[str] = []

            for var in column_labels:
                m = func(data_chunk[var].to_numpy())
                row_text.append(f"{m:.2f}")

            cell_text.append(row_text)

        ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=column_labels,
        )

        # Check for ICMEs
        contains_icme, icme_count = check_for_icmes(data_chunk, icmecat)

        ax.text(
            -0.1,
            0,
            f"Contains ICME: {contains_icme}\n" + f"ICME count: {icme_count}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

        plt.tight_layout()
        plt.show()


def check_for_icmes(data: pl.DataFrame, icme_list: pl.DataFrame) -> Tuple[bool, int]:

    data_start = data["UTC"][0]
    data_end = data["UTC"][-1]

    time_range = TimeRange(data_start, data_end)

    # Loop through ICMES
    count: int = 0
    for row in icme_list.iter_rows(named=True):
        icme_start = row["Start Time"]
        icme_end = icme_start + dt.timedelta(hours=row["Duration"])
        icme_range = TimeRange(icme_start, icme_end)

        if (
            icme_start in time_range
            or icme_end in time_range
            or data_start in icme_range
        ):

            count += 1

    return count > 0, count


def create_plot(ax: Axes, data) -> None:

    variables = ["|B|", "Br", "Bt", "Bn"]
    colours = ["black", "#D55E00", "#009E73", "#0072B2"]

    for var, colour in zip(variables, colours):
        ax.plot(data["UTC"], data[var + " [nT]"], color=colour, label=var, lw=0.5)


def get_times_near_mercury(
    time_range: TimeRange,
    data: pl.DataFrame,
    spacecraft: str = "Solar Orbiter",
    distance_limits: Tuple[u.Quantity, u.Quantity] = (0.3075 * u.au, 0.4667 * u.au),
    latitude_limits: u.Quantity = 3.38 * u.deg,  # maximum +/-
    trajectory_resolution: u.Quantity = 1 * u.hour,
) -> List[pl.DataFrame]:
    """
    Use spiceypy and some constraints to determine times where spaecraft were
    'near Mercury'
    """

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

    with spice_client.KernelPool():

        times: Sequence[dt.datetime] = [
            t.center.to_datetime()
            for t in time_range.split(
                int((time_range.end - time_range.start) / trajectory_resolution)
            )
        ]

        ets = spice.datetime2et(times)

        positions, _ = spice.spkpos(spacecraft, ets, "J2000", "NONE", "SUN")

    positions *= u.km

    skycoords = SkyCoord(
        CartesianRepresentation(positions.T),
        obstime=times,
        frame="icrs",
    ).transform_to(frames.HeliographicStonyhurst)

    assert isinstance(skycoords.radius, u.Quantity)
    assert isinstance(skycoords.lon, u.Quantity)
    assert isinstance(skycoords.lat, u.Quantity)

    positions_table = pl.DataFrame(
        {
            "UTC": times,
            "Radius [au]": skycoords.radius.to(u.au),
            "Longitude [deg]": skycoords.lon.to(u.deg),
            "Latitude [deg]": skycoords.lat.to(u.deg),
        }
    )

    # Filter only to times within critera
    positions_table = positions_table.filter(
        (pl.col("Radius [au]").is_between(*distance_limits))
        & (pl.col(["Latitude [deg]"]).abs() <= latitude_limits)
    )

    # We only want the time intervals for these 'near Mercury' stints.
    positions_table = positions_table.with_columns(
        (pl.col("UTC") - pl.col("UTC").shift(1)).alias("Time Step")
    )

    jump_indices = np.where(
        positions_table["Time Step"]
        > dt.timedelta(hours=trajectory_resolution.to(u.hour).value)
    )[0]

    start_indices = np.concatenate(([0], jump_indices)).tolist()
    end_indices = np.concatenate(
        (jump_indices - 1, [len(positions_table) - 1])
    ).tolist()

    time_intervals = [
        TimeRange(positions_table["UTC"][s], positions_table["UTC"][e])
        for s, e in zip(start_indices, end_indices)
    ]

    near_mercury_data: List[pl.DataFrame] = []

    for tr in time_intervals:

        filtered_data = data.filter(
            pl.col("UTC").is_between(tr.start.to_datetime(), tr.end.to_datetime())
        )
        near_mercury_data.append(filtered_data)

    return near_mercury_data


if __name__ == "__main__":
    main()
