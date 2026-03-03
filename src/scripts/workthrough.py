"""
Just a script to work through the required components for modelling in this work.
"""

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from sunpy.time import TimeRange

from vswim.data import MAGData, get_solar_orbiter_data
from vswim.modelling import SolarWindModel

SEED = 1785
N_INDUCING_POINTS = 5

tf.random.set_seed(SEED)

# To train a model, we need an interval of data within which we create artificial gaps

# Make Data object
mag: MAGData = MAGData(
    get_solar_orbiter_data(TimeRange("2021-01-01", "2021-01-01 01:00")),
    metadata={"Spacecraft": "Solar Orbiter"},
)

X: NDArray = mag.data["UTC"].to_numpy().reshape(-1, 1)
Y: NDArray = mag.data["|B| [nT]"].to_numpy().reshape(-1, 1).astype("float64")

# For now we include the full dataset. This model class has functionality to
# separate a training and testing set.
model = SolarWindModel.build(
    input=X,
    output=Y,
    n_inducing_points=10,
    seed=SEED,
)

print(model.data)


# Xplot = np.linspace(0, 1, 100)[:, None]
#
# f_mean, f_var = model.predict_f(Xplot, full_cov=False)
# y_mean, y_var = model.predict_y(Xplot)
#
# f_lower = f_mean - 1.96 * np.sqrt(f_var)
# f_upper = f_mean + 1.96 * np.sqrt(f_var)
# y_lower = y_mean - 1.96 * np.sqrt(y_var)
# y_upper = y_mean + 1.96 * np.sqrt(y_var)
#
# plt.plot(X, Y, "kx", mew=2, label="input data")
# plt.plot(Xplot, f_mean, "-", color="C0", label="mean")
# plt.plot(Xplot, f_lower, "--", color="C0", label="f 95% confidence")
# plt.plot(Xplot, f_upper, "--", color="C0")
# plt.fill_between(Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1)
# plt.plot(Xplot, y_lower, ".", color="C0", label="Y 95% confidence")
# plt.plot(Xplot, y_upper, ".", color="C0")
# plt.fill_between(Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1)
# plt.legend()
#
# plt.show()
