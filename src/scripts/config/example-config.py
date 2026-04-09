import datetime as dt

from gpflow import Parameter
from gpflow.kernels import Periodic, RationalQuadratic, SquaredExponential

from mvswim.modelling import GapGenerator

kernel = RationalQuadratic() + Periodic(
    base_kernel=SquaredExponential(),
    period=Parameter(27 * 24 * 3600, trainable=False),  # Seconds
)

CONFIG = {
    "Seed": 1785,
    "Data": {
        "Spacecraft": ["Solar Orbiter"],
        "Filter": {
            "Filter Resolution": dt.timedelta(hours=1),
            "Heliocentric Distance [AU]": (0.3075, 0.4667),
            "Latitude [deg]": 3.38,  # maxiumum +/-
        },
        "Downsample": {
            # GPR models scale dramatically with the amount of data used. We
            # downsample from our minute resolution data to achieve better
            # performance.
            "Enabled": True,
            "Frequency": "30m",
        },
    },
    "Model": {
        "Inducing Point Fraction": 0.05,  # Fraction of data to fit to (0-1)
        "Gap Generator": GapGenerator.from_gaussian(
            gap_size_mean=4 * 60,  # minutes
            gap_size_std=60,
            gap_interval_mean=5 * 60,
            gap_interval_std=60,
        ),
        "Kernel": kernel,
    },
}
