"""
We need to iterate through a list of MESSENGER bow shock and magnetopause
crossings to find all lengths of time in the solar wind. These can be long
segments of the orbit, but also short spans during multiple crossing events.

Time in the solar wind is the time between consecutive BS_OUT and BS_IN labels.
"""

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import requests
from hermpy.data import CrossingList, InstantEventList

DATA_DIRECTORY = Path(__file__).parent.parent.parent / "data/"
FIGURE_DIRECTORY = Path(__file__).parent.parent.parent / "figures/"

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

# Make a new crossing list from csv file
crossings: InstantEventList = CrossingList.from_csv(
    crossing_list_path, time_column="Time"
)

# We define a set of indices describing the start of each stint in the solar
# wind. These are times of outbound bow shocks, which are followed by inbound
# bow shocks.
inbound_bow_shock_indices = np.where(crossings.table["Label"] == "BS_IN")[0]
outbound_bow_shock_indices = np.where(crossings.table["Label"] == "BS_OUT")[0]

solar_wind_start_indices = [
    int(outbound_bow_shock_indices[i])
    for i in range(len(outbound_bow_shock_indices))
    if outbound_bow_shock_indices[i] + 1 in inbound_bow_shock_indices
]

solar_wind_end_indices = [
    int(inbound_bow_shock_indices[i])
    for i in range(len(inbound_bow_shock_indices))
    if inbound_bow_shock_indices[i] - 1 in outbound_bow_shock_indices
]
# The last outbound doesn't have an inbound, so we remove the
# last start index.
solar_wind_start_indices = np.array(solar_wind_start_indices)

# Create a dictionary to hold information about each solar wind region
solar_wind_stints: dict[str, Any] = {
    "Start Time": crossings.table["Time"][solar_wind_start_indices],
    "End Time": crossings.table["Time"][solar_wind_end_indices],
}
solar_wind_stints["Length"] = (
    solar_wind_stints["End Time"] - solar_wind_stints["Start Time"]
)

# It is also simple to find the gaps in continuous solar wind measurements as
# the compliment of these. We look at the times from the ends to the starts.
gaps: dict[str, Any] = {
    "Start Time": solar_wind_stints["End Time"][:-1],
    "End Time": solar_wind_stints["Start Time"][1:],
}
gaps["Length"] = gaps["End Time"] - gaps["Start Time"]

# There is one outlier here due to a data gap lining up with a data gap. We
# should be able to work around this, but need to check if there is a data gap
# interval within each solar wind stint. If so, we just split the stint into
# # two.
# print(np.where(solar_wind_stints["Length"].to_value("hour") > 12)[0])
#
# print(solar_wind_stints["Start Time"][1775])
# print(solar_wind_stints["Length"][1775].to_value("hour"))

# PLOTTING
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

sw_ax, gap_ax = axes

sw_ax.set_title("MESSENGER's Solar Wind Stints")
sw_ax.set_xlabel("Consecutive Hours Observed")
sw_ax.set_ylabel("Number of Solar Wind Excursions")

for ax in axes:
    ax.margins(x=0)
    ax.set_yscale("log")

bin_size = 0.5  # hours
bins = np.arange(0, 12 + bin_size, bin_size).tolist()

sw_ax.hist(solar_wind_stints["Length"].to_value("hour"), bins=bins, color="black")
gap_ax.hist(gaps["Length"].to_value("hour"), bins=bins, color="black")

gap_ax.set_title("MESSENGER's Solar Wind Gaps")
gap_ax.set_xlabel("Consecutive Hours without Observations")
gap_ax.set_ylabel("Number of Gaps")

plt.tight_layout()
plt.savefig(FIGURE_DIRECTORY / "solar-wind-stints.pdf", format="pdf")
