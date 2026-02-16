"""
A script to investigate the availability of many spacecraft for use with this
work.
"""

import datetime as dt
from pathlib import Path
from typing import Any, Dict

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.units import Quantity
from hermpy.net import ClientSPICE
from matplotlib.dates import MonthLocator

# Define the critera with which we look within
HELIOCENTRIC_DISTANCE_BOUNDS = (0.3 * u.au, 0.5 * u.au)
FIGURE_DIRECTORY = Path(__file__).parent.parent / "figures/"

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
            "PATTERNS": ["solo_ANC_soc-orbit_20200210-20301118_L000_V0_00001_V01.bsp"],
        },
        # BepiColombo
        "BepiColombo Positions": {
            "BASE": "http://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/",
            "DIRECTORY": "kernels/spk/",
            "PATTERNS": [
                "de432s.bsp",
                "earthstns_itrf93_201023.bsp",
                "earthstns_jaxa_20230905.bsp",
                "estrack_v04.bsp",
                "bc_sci_v02.bsp",
                "bc_mmo_struct_v01.bsp",
                "bc_mmo_cruise_v02.bsp",
                "bc_mtm_struct_v06.bsp",
                "bc_mtm_cruise_v02.bsp",
                "bc_mpo_cog_v03.bsp",
                "bc_mpo_cog_00220_20181118_20260327_v01.bsp",
                "bc_mpo_struct_v12.bsp",
                "bc_mpo_schulte_vector_v01.bsp",
                "bc_mpo_prelaunch_v01.bsp",
                "bc_mpo_fcp_00220_20181020_20270407_v01.bsp",
            ],
        },
        "BepiColombo Frames": {
            "BASE": "http://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/",
            "DIRECTORY": "kernels/fk/",
            "PATTERNS": [
                "bc_mpo_v39.tf",
                "bc_mtm_v12.tf",
                "bc_mmo_v14.tf",
                "bc_ops_v02.tf",
                "bc_sci_v12.tf",
                "bc_dsk_surfaces_v03.tf",
                "rssd0004.tf",
                "earth_topo_201023.tf",
                "earthstns_jaxa_20230905.tf",
                "earthfixeditrf93.tf",
                "estrack_v04.tf",
            ],
        },
        # "MARINER 10": {
        #     "BASE": "https://naif.jpl.nasa.gov/pub/naif/M10/",
        #     "DIRECTORY": "kernels/spk/",
        #     "PATTERNS": ["M10_archive_1.bsp"],
        # },
        # "Helios 1/2": {
        #     "BASE": "https://naif.jpl.nasa.gov/pub/naif/HELIOS/",
        #     "DIRECTORY": "kernels/spk/",
        #     "PATTERNS": ["???????_???????_?????_?????.bsp"],
        # },
        "PSP": {
            "BASE": "https://spdf.gsfc.nasa.gov/pub/data/psp/",
            "DIRECTORY": "ephemeris/spice/ephemerides/",
            "PATTERNS": ["spp_nom_20180812_20300101_v043_PostV7.bsp"],
        },
    }
)

spacecraft_info: Dict[str, Dict[str, Any]] = {
    # "Helios 2": {
    #     "ID": "Helios 2",
    #     "Time Range": (dt.datetime(1976, 1, 16), dt.datetime(1979, 12, 22)),
    # },
    # "Helios 1": {
    #     "ID": "Helios 1",
    #     "Time Range": (dt.datetime(1974, 12, 11), dt.datetime(1981, 2, 17)),
    # },
    "BepiColombo": {
        "ID": "BEPICOLOMBO MPO",
        "Time Range": (dt.datetime(2018, 10, 30), dt.datetime.today()),
    },
    "Solar Orbiter": {
        "ID": "Solar Orbiter",
        "Time Range": (dt.datetime(2020, 2, 11), dt.datetime.today()),
    },
    "Parker Solar Probe": {
        "ID": "Parker Solar Probe",
        "Time Range": (dt.datetime(2018, 8, 13), dt.datetime.today()),
    },
}

# With the positions of the spacecraft we have, we can caluclate the
# heliocentric distance.
with spice_client.KernelPool():
    for key, info in spacecraft_info.items():

        info["Time"] = [
            info["Time Range"][0] + i * dt.timedelta(days=1)
            for i in range((info["Time Range"][1] - info["Time Range"][0]).days)
        ]
        ets = spice.datetime2et(info["Time"])

        info["Positions"] = (
            spice.spkpos(info["ID"], ets, "J2000", "NONE", "SUN")[0] * u.km
        )

        info["Heliocentric Distance"] = np.linalg.norm(info["Positions"], axis=1)

        assert isinstance(info["Positions"], Quantity)
        assert isinstance(info["Heliocentric Distance"], Quantity)


# Contruct our plot

_, ax = plt.subplots(figsize=(6, 2))

for i, (_, info) in enumerate(spacecraft_info.items()):

    is_within_heliocentric_distance_bounds = (
        info["Heliocentric Distance"] > HELIOCENTRIC_DISTANCE_BOUNDS[0]
    ) & (info["Heliocentric Distance"] <= HELIOCENTRIC_DISTANCE_BOUNDS[1])

    values = np.where(is_within_heliocentric_distance_bounds, i, np.nan)

    ax.plot(info["Time"], values, lw=10, solid_capstyle="butt")


# Replace ticks with spacecraft labels
ax.set_yticks(range(len(spacecraft_info)), spacecraft_info.keys())

ax.set_title(
    f"{HELIOCENTRIC_DISTANCE_BOUNDS[0]} < "
    r"$R_{\rm H}$" + f" ≤ {HELIOCENTRIC_DISTANCE_BOUNDS[1]}"
)

ax.margins(x=0, y=0)
ax.set_ylim(-0.5, 3 - 0.5)
ax.tick_params(axis="x", rotation=-90)
ax.xaxis.set_minor_locator(MonthLocator())

plt.tight_layout()
plt.savefig(FIGURE_DIRECTORY / "spacecraft-availability.pdf", format="pdf")
