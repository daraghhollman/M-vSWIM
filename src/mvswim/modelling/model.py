from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
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
        time_scaler: TimeScaler,
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
        X = time_scaler.time_to_numeric(input)
        Y = output

        kernel = _build_kernel(time_scaler)

        # Use K-Means Clustering to determine good inducing points
        kmeans = MiniBatchKMeans(
            n_clusters=n_inducing_points, random_state=seed, n_init="auto"
        )
        kmeans.fit(X)
        inducing_points = kmeans.cluster_centers_

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

    def log(self, s: str) -> None:

        # Print the statement
        print(s)

        # Log the statement to a file
        with open(self.log_directory / "log", "a") as f:
            f.write(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S  ") + s + "\n")

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

        start_time = dt.datetime.now()
        self.log(f"Training started at: {start_time}")

        self.optimiser.minimize(
            self.model.training_loss,
            self.model.trainable_variables,
            step_callback=monitor,
        )

        end_time = dt.datetime.now()
        self.log(f"Training ended at: {end_time}")

        self.log(f"Total training time: {end_time - start_time}")

        return

    def test_performance(
        self, testing_x: NDArray, testing_y: NDArray
    ) -> Dict[str, Dict[str, NDArray]]:

        self.log(f"TESTING")
        # We need to scale our inputs
        testing_x = self.time_scaler.time_to_numeric(testing_x)

        y_mean, _ = self.model.predict_y(testing_x)

        # Define a list of metrics and labels
        metric_labels: List[str] = ["RMSE", "MAE", "R Squared"]
        metric_functions: List[Callable] = [
            root_mean_squared_error,
            mean_absolute_error,
            r2_score,
        ]

        metrics: Dict[str, Any] = {}

        label: str
        function: Callable
        for label, function in zip(metric_labels, metric_functions):

            # Each of these are in the form: y_true, y_pred
            metric_values = function(testing_y, y_mean)

            metrics.update(
                {label: {"Mean": np.mean(metric_values), "SD": np.std(metric_values)}}
            )

            self.log(
                f"    {label}: {np.mean(metric_values):.3f} +/- {np.std(metric_values):.3f}"
            )

        return metrics

    def get_training_loss(self) -> Tensor:
        return self.model.training_loss()

    def info(self) -> None:
        """
        Prints info about the trained parameters.
        """
        gpflow.utilities.print_summary(self.model)

    def quicklook(self, testing_data: None | Tuple[NDArray, ...] = None) -> None:
        x_range = np.linspace(0, 1, 1000)[:, None]

        f_mean, f_var = self.model.predict_f(x_range, full_cov=False)
        y_mean, y_var = self.model.predict_y(x_range)

        y_lower = y_mean - 1.96 * np.sqrt(y_var)
        y_upper = y_mean + 1.96 * np.sqrt(y_var)

        _, ax = plt.subplots(figsize=(12, 8))

        ax.scatter(
            self.time_scaler.numeric_to_time(self.data[0]),
            self.data[1],
            color="black",
            marker=".",
            label="input data",
        )

        if testing_data is not None:
            ax.scatter(
                *testing_data, color="indianred", marker=".", label="testing data"
            )

        x_range = self.time_scaler.numeric_to_time(x_range)
        ax.plot(x_range, f_mean, "-", color="C0", label="Prediction Mean")
        ax.plot(x_range, y_lower, "-", color="C0", label="95% confidence")
        ax.plot(x_range, y_upper, "-", color="C0")
        ax.fill_between(
            x_range[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
        )
        ax.legend()

        fig_path = (
            self.log_directory
            / f"figures/training-{dt.datetime.now().strftime('%H:%M:%S')}.pdf"
        )

        # Ensure directory exits
        os.makedirs(fig_path.parent, exist_ok=True)

        self.log(f"Saving figure to {fig_path}")

        plt.savefig(fig_path, format="pdf")


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
