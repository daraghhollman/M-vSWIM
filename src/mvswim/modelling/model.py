from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import astropy.units as u
import gpflow
import matplotlib.pyplot as plt
import numpy as np
from gpflow import Parameter
from gpflow.kernels import Kernel, Periodic, RationalQuadratic
from gpflow.models import SGPR
from gpflow.monitor import Monitor, MonitorTaskGroup, ScalarToTensorBoard
from gpflow.optimizers import Scipy
from keras.optimizers import Optimizer
from numpy.typing import NDArray
from tensorflow import Tensor

from mvswim.constants import CARRINGTON_ROTATION
from mvswim.scalling import TimeScaler

__all__ = [
    "SolarWindModel",
]


# Define a class to hold the model and associated functions
@dataclass
class SolarWindModel:
    model: SGPR
    data: Tuple[NDArray[np.datetime64], NDArray[Any]]
    optimiser: Optimizer
    time_scaler: TimeScaler  # Store scaler on model for later inverse transforms
    seed: int
    log_directory: Path

    @classmethod
    def build(
        cls,
        input: NDArray[np.datetime64],
        output: NDArray[Any],
        n_inducing_points: int,
        seed: int,
        log_directory: Path,
    ) -> SolarWindModel:
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

        opt = Scipy()

        return cls(
            model=gpmodel,
            data=(X, Y),
            optimiser=opt,
            time_scaler=time_scaler,
            seed=seed,
            log_directory=log_directory,
        )

    def train_model(self) -> None:
        """
        Perform iterations of training.
        """

        loss_monitor = ScalarToTensorBoard(
            str(self.log_directory / "training-loss"),
            self.get_training_loss,
            "Training Loss",
        )
        task_group = MonitorTaskGroup(loss_monitor, period=3)
        monitor = Monitor(task_group)

        # We can do some other logging ourselves
        # Create training log file
        log_file = self.log_directory / "log"
        with open(log_file, "a") as f:
            start_time = dt.datetime.now()
            f.write(f"Training started at: {start_time}\n")

        self.optimiser.minimize(
            self.model.training_loss,
            self.model.trainable_variables,
            step_callback=monitor,
        )

        with open(log_file, "a") as f:
            end_time = dt.datetime.now()
            f.write(f"Training finished at: {end_time}\n")

        with open(log_file, "a") as f:
            f.write(f"Total training time: {end_time - start_time}\n")

    def get_training_loss(self) -> Tensor:
        return self.model.training_loss()

    def info(self) -> None:
        """
        Prints info about the trained parameters.
        """
        gpflow.utilities.print_summary(self.model)

    def quicklook(self) -> None:
        Xplot = np.linspace(0, 1, 1000)[:, None]

        f_mean, f_var = self.model.predict_f(Xplot, full_cov=False)
        y_mean, y_var = self.model.predict_y(Xplot)

        f_lower = f_mean - 1.96 * np.sqrt(f_var)
        f_upper = f_mean + 1.96 * np.sqrt(f_var)
        y_lower = y_mean - 1.96 * np.sqrt(y_var)
        y_upper = y_mean + 1.96 * np.sqrt(y_var)

        plt.scatter(*self.data, color="black", marker=".", label="input data")
        plt.plot(Xplot, f_mean, "-", color="C0", label="mean")
        plt.plot(Xplot, f_lower, "--", color="C0", label="f 95% confidence")
        plt.plot(Xplot, f_upper, "--", color="C0")
        plt.fill_between(
            Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1
        )
        plt.plot(Xplot, y_lower, "-", color="C0", label="Y 95% confidence")
        plt.plot(Xplot, y_upper, "-", color="C0")
        plt.fill_between(
            Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
        )
        plt.legend()

        plt.show()


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

    # We will capture the general shape with a rational quadratic
    trend = RationalQuadratic()

    return trend + RationalQuadratic() * periodic_component
