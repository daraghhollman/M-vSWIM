import datetime as dt

from mvswim.modelling import GapGenerator

CONFIG = {
    "Seed": 1785,
    "Data": {
        "Spacecraft": ["Solar Orbiter"],
        "Filter": {
            "Filter Resolution": dt.timedelta(hours=1),
            "Heliocentric Distance [AU]": (0.3075, 0.4667),
            "Latitude [deg]": 3.38,
        },
    },
    "Model": {
        "Inducing Points": 10,
        "Gap Generator": GapGenerator.from_gaussian(
            gap_size_mean=60,  # minutes
            gap_size_std=10,
            gap_interval_mean=60,
            gap_interval_std=10,
        ),
    },
}
