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
        "Gap Generator": GapGenerator.from_normal_distributions(2 * 60, 30, 4 * 60, 30),
    },
}
