import datetime as dt

from gpflow import Parameter
from gpflow.kernels import Periodic, RationalQuadratic, SquaredExponential

from mvswim.modelling import GapGenerator

kernel = RationalQuadratic() + Periodic(
    base_kernel=SquaredExponential(),
    # Period in data cadence units
    period=Parameter(27 * 24),
)

CONFIG = {
    "Seed": 1785,
    "Data": {
        "Spacecraft": ["Solar Orbiter"],
        "Filter": {
            "Filter Resolution": dt.timedelta(hours=1),
            "Heliocentric Distance [AU]": (0.3075, 0.4667),
            "Latitude [deg]": 3.38,
        },
        "Downsample": {
            "Enabled": True,
            "Frequency": "1h",
        },
    },
    "Model": {
        "Inducing Point Fraction": 0.01,  # Fraction of data to fit to (0-1)
        "Gap Generator": GapGenerator.from_gaussian(
            gap_size_mean=60,  # minutes
            gap_size_std=10,
            gap_interval_mean=60,
            gap_interval_std=10,
        ),
        "Kernel": kernel,
    },
}
