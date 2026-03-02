"""
Just a script to work through the required components for modelling in this work.
"""

import gpflow
import matplotlib.pyplot as plt
import numpy as np
from gpflow.kernels import RationalQuadratic, SquaredExponential
from gpflow.models import SGPR
from numpy.typing import NDArray
from sunpy.time import TimeRange

from vswim.data import MAGData, get_solar_orbiter_data
from vswim.modelling import TimeScaler

SEED = 1785
N_INDUCING_POINTS = 5

# To train a model, we need an interval of data within which we create artificial gaps

# Make Data object
mag: MAGData = MAGData(
    get_solar_orbiter_data(TimeRange("2021-01-01", "2021-01-01 01:00")),
    metadata={"Spacecraft": "Solar Orbiter"},
)

X: NDArray = mag.data["UTC"].to_numpy().reshape(-1, 1)

# X is a datetime object which we must first convert to being numerical.
# Additionally, as GP models are based on distance between data, we want to use
# small numerical values to aid in computation times. For this, we scale the
# time between 0 and 1.
time_scaler = TimeScaler(X)
X = time_scaler.time_to_numeric(X)

Y: NDArray = mag.data["|B| [nT]"].to_numpy().reshape(-1, 1).astype("float64")

# Choose inducing points randomly
rng = np.random.default_rng(SEED)
inducing_points = rng.choice(X, size=N_INDUCING_POINTS, replace=False)
k = RationalQuadratic()
model = SGPR((X, Y), kernel=k, inducing_variable=inducing_points)

opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

Xplot = np.linspace(0, 1, 100)[:, None]

f_mean, f_var = model.predict_f(Xplot, full_cov=False)
y_mean, y_var = model.predict_y(Xplot)

f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)

plt.plot(X, Y, "kx", mew=2, label="input data")
plt.plot(Xplot, f_mean, "-", color="C0", label="mean")
plt.plot(Xplot, f_lower, "--", color="C0", label="f 95% confidence")
plt.plot(Xplot, f_upper, "--", color="C0")
plt.fill_between(Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1)
plt.plot(Xplot, y_lower, ".", color="C0", label="Y 95% confidence")
plt.plot(Xplot, y_upper, ".", color="C0")
plt.fill_between(Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1)
plt.legend()

plt.show()
