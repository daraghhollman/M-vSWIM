"""
Use MESSENGER ephemeris data to determine orbit number. We create a look-up
table for the start times of each orbit - and then query that.
"""

import datetime as dt
import os
from pathlib import Path

import numpy as np
import polars as pl
import spiceypy as spice
from hermpy.net import ClientSPICE
from scipy.signal import find_peaks

ORBIT_TABLE_PATH = (
    Path(__file__).parent.parent.parent / "data/messenger-orbit-table.csv"
)
MISSION_START = dt.datetime(2011, 3, 18)
MISSION_END = dt.datetime(2015, 4, 30)


def create_orbit_table():

    # Load MESSENGER ephemeris data
    spice_client = ClientSPICE()

    # Get positions for whole mission.
    temporal_resolution = dt.timedelta(minutes=10)

    query_times = [
        MISSION_START + i * temporal_resolution
        for i in range(int((MISSION_END - MISSION_START) / temporal_resolution))
    ]

    with spice_client.KernelPool():
        positions, _ = spice.spkpos(
            "MESSENGER", spice.datetime2et(query_times), "MSGR_MSO", "NONE", "Mercury"
        )

    distances = np.linalg.norm(positions, axis=1)

    # Find periapsis points
    new_orbit_indices, _ = find_peaks(-distances)

    new_orbit_times = np.array(query_times)[new_orbit_indices]

    orbit_table = pl.DataFrame(
        {
            # Index from one as time before the first periapsis can be referred
            # to as orbit 0.
            "Orbit Number": np.arange(1, len(new_orbit_indices) + 1),
            "Start Time": new_orbit_times.tolist(),
        }
    )

    orbit_table.with_columns(pl.col("Start Time").cast(pl.String))

    orbit_table.write_csv(ORBIT_TABLE_PATH)


def get_orbit_number(time: dt.datetime):

    # If the table doesn't exist we must generate it
    if not os.path.exists(ORBIT_TABLE_PATH):
        create_orbit_table()

    orbit_table = pl.read_csv(ORBIT_TABLE_PATH, try_parse_dates=True)

    # If the query time is before the first orbit start, we are in orbit 0.
    if time < orbit_table["Start Time"][0]:
        return 0

    # The negative time difference closest to 0 will give us the index of our current
    # orbit.
    time_differences = orbit_table["Start Time"] - time

    # Filter to only negative differences (times before or equal to `time`),
    # then find the index of the one closest to 0 (i.e., the largest negative value).
    closest_index = (
        time_differences.to_frame("diff")
        .with_row_index("index")
        .filter(pl.col("diff") <= pl.duration(seconds=0))
        .sort("diff", descending=True)
        .row(0)[0]
    )

    return orbit_table["Orbit Number"][closest_index]
