from dataclasses import dataclass
from typing import Any, Tuple

import astropy.units as u
import numpy as np
import tensorflow as tf
from gpflow import Parameter
from gpflow.kernels import Kernel, Periodic, RationalQuadratic
from gpflow.models import SGPR, GPModel
from keras.optimizers import Adam, Optimizer
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

from vswim.constants import CARRINGTON_ROTATION

__all__ = [
    "TimeScaler",
    "SolarWindModel",
]


class TimeScaler:

    def __init__(self, fit_data: NDArray) -> None:

        self._fit_data = fit_data
        self._scaler = self.init_transform()

    def init_transform(self) -> MinMaxScaler:
        scaler = MinMaxScaler()
        scaler.fit(self._fit_data)

        return scaler

    def time_to_numeric(self, data: NDArray) -> NDArray:
        """
        Converts an array of np.datetime64 to numeric values scaled between 0
        and 1.
        """

        return self._scaler.transform(data.astype("int64"))

    def numeric_to_time(self, data: NDArray) -> NDArray:
        """
        Performs the reverse operation of 'time_to_numeric', converting numeric
        values scaled between 0 and 1, back to their original time values.
        """

        return self._scaler.inverse_transform(data).astype("datetime64[ns]")

    def scale_duration(self, duration_seconds: float) -> float:
        """
        Converts a physical duration (in seconds) into the scaled [0, 1] time
        units used by the model. Durations scale by the data range only —
        no offset is applied, unlike point values.
        """
        data_range_ns = float(self._scaler.data_range_[0])
        duration_ns = duration_seconds * 1e9
        return duration_ns / data_range_ns


# Define a class to hold the model and associated functions
@dataclass
class SolarWindModel:
    model: GPModel
    data: Tuple[NDArray[np.datetime64], NDArray[Any]]
    optimiser: Optimizer
    seed: int
    time_scaler: TimeScaler  # Store scaler on model for later inverse transforms

    @classmethod
    def build(
        cls,
        input: NDArray[np.datetime64],
        output: NDArray[Any],
        n_inducing_points: int,
        seed: int,
    ) -> "SolarWindModel":
        """
        We choose to define how our model is constructed here so that the
        inital choices are fixed. Any user can override any attributes or
        methods as they see fit. i.e. changing an optimiser, or how the
        training call works.
        """

        # X is a datetime object which we must first convert to being numerical.
        # Additionally, as GP models are based on distance between data, we want to use
        # small numerical values to aid in computation times. For this, we scale the
        # time between 0 and 1.
        time_scaler = TimeScaler(input)
        X = time_scaler.time_to_numeric(input)
        Y = output

        kernel = _build_kernel(time_scaler)

        # Currently this is a random choice, but we can do something more
        # sophisticated such as k-means.
        rng = np.random.default_rng(seed)
        inducing_points = rng.choice(X, size=n_inducing_points, replace=False)

        gpmodel = SGPR((X, Y), kernel=kernel, inducing_variable=inducing_points)

        opt = Adam()

        return cls(
            model=gpmodel,
            data=(X, Y),
            optimiser=opt,
            seed=seed,
            time_scaler=time_scaler,
        )

    def train_model(self, n_iterations: int) -> None:
        """
        Perform iterations of training.
        """
        pass


def _build_kernel(time_scaler: TimeScaler) -> Kernel:
    """
    Constructs the solar wind kernel with physically-meaningful initial
    parameter values expressed in scaled time units.
    """
    # We expect a periodic component with period roughly equal to the time it
    # takes for the same part of the sun to be subsolar to Mercury again - note
    # that this is slightly longer than a solar rotation.

    # Scale the physical period into [0,1] time units
    scaled_period = time_scaler.scale_duration(CARRINGTON_ROTATION.to(u.second).value)

    periodic_component = Periodic(
        base_kernel=RationalQuadratic(),
        period=Parameter(scaled_period, trainable=True),
    )

    # We may also expect a trend
    return periodic_component
