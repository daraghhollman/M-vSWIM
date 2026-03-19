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
import polars as pl
import tensorflow as tf
from gpflow import Parameter
from gpflow.kernels import (
    Kernel,
    Linear,
    Periodic,
    RationalQuadratic,
    SquaredExponential,
)
from gpflow.models import SGPR
from gpflow.monitor import Monitor, MonitorTaskGroup, ScalarToTensorBoard
from gpflow.optimizers import Scipy
from gpflow.utilities import deepcopy
from keras.optimizers import Optimizer
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from tensorflow import Tensor

from mvswim.constants import CARRINGTON_ROTATION
from mvswim.scalling import TimeScaler

__all__ = [
    "SolarWindModel",
    "plot_from_training_data",
]


# Define a class to hold the model and associated functions
@dataclass
class SolarWindModel:
    model: SGPR
    data: Tuple[NDArray[np.datetime64], NDArray[Any]]
    kernel: Kernel
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
            kernel=kernel,
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

    def visualise_kernel(self) -> None:
        # Visualise kernel
        fig = _plot_kernel(self.kernel)

        os.makedirs(self.log_directory / "figures")

        fig.savefig(self.log_directory / f"figures/kernel.pdf")

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

        # Get model predictions for this time range
        y_mean, _ = self.model.predict_y(testing_x)

        # Get linear interpolation for this time range
        linear_y = _interpolate_continuous_chunks(testing_x, testing_y)

        # Define a list of metrics and labels
        metric_labels: List[str] = ["RMSE", "MAE", "R Squared"]
        metric_functions: List[Callable] = [
            root_mean_squared_error,
            mean_absolute_error,
            r2_score,
        ]

        model_metrics: Dict[str, Any] = {}
        linear_metrics: Dict[str, Any] = {}

        label: str
        function: Callable
        for label, function in zip(metric_labels, metric_functions):

            # Each of these are in the form: y_true, y_pred
            metric = function(testing_y, y_mean)
            model_metrics.update({label: metric})
            self.log(f"    Model {label}: {metric}")

            metric = function(testing_y, linear_y)
            linear_metrics.update({label: metric})
            self.log(f"    LI {label}: {metric}")

        return {"Model": model_metrics, "Linear Interpolation": linear_metrics}

    def get_training_loss(self) -> Tensor:
        return self.model.training_loss()

    def info(self) -> None:
        """
        Prints info about the trained parameters.
        """
        gpflow.utilities.print_summary(self.model)

    def quicklook(self, testing_data: None | Tuple[NDArray, ...] = None) -> None:
        x_range = np.linspace(0, 1, 1000)[:, None]

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

            linear_y = _interpolate_continuous_chunks(
                self.time_scaler.time_to_numeric(testing_data[0]), testing_data[1]
            )

            ax.scatter(
                testing_data[0],
                linear_y,
                color="cornflowerblue",
                label="Linear Interpolation",
                marker=".",
            )

        x_range = self.time_scaler.numeric_to_time(x_range)
        ax.plot(x_range, y_mean, "-", color="C0", label="Prediction Mean")
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

        # We also want to save the data used to create the plot, so it can be
        # investigated deeper.
        if testing_data is not None:

            data_path = (
                self.log_directory
                / f"data/training-{dt.datetime.now().strftime('%H:%M:%S')}.npz"
            )
            os.makedirs(data_path.parent, exist_ok=True)
            self.log(f"Writing figure data to {data_path}")

            np.savez(
                data_path,
                x_train=self.time_scaler.numeric_to_time(self.data[0]).squeeze(),
                y_train=self.data[1],
                x_test=testing_data[0],
                y_test=testing_data[1],
                x_range=x_range,
                y_mean=y_mean,
                y_upper=y_upper,
                y_lower=y_lower,
                linear_interpolation=linear_y,
            )


def plot_from_training_data(data_path: Path) -> Tuple[Figure, Axes]:
    """
    Our model generates quicklook plots when training. These are nice for a
    quicklook, but we also save the data with the logs to look more closely.
    This function takes those data files as input, and creates a matplotlib
    plot to view them.
    """

    loaded_data = np.load(data_path)

    data = {}
    for key in loaded_data:
        data[key] = loaded_data[key].squeeze()

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(
        data["x_train"],
        data["y_train"],
        color="black",
        marker=".",
        label="Training Data",
    )

    ax.scatter(
        data["x_test"],
        data["y_test"],
        color="indianred",
        marker=".",
        label="Testing Data",
    )

    ax.scatter(
        data["x_test"],
        data["linear_interpolation"],
        color="cornflowerblue",
        label="Linear Interpolation",
        marker=".",
    )

    ax.plot(data["x_range"], data["y_mean"], "-", color="C0", label="Prediction Mean")
    ax.plot(data["x_range"], data["y_upper"], "-", color="C0", label="95% confidence")
    ax.plot(data["x_range"], data["y_lower"], "-", color="C0")
    ax.fill_between(
        data["x_range"],
        data["y_lower"].squeeze(),
        data["y_upper"].squeeze(),
        color="C0",
        alpha=0.1,
    )
    ax.legend()

    return fig, ax


def _build_kernel(time_scaler: TimeScaler) -> Kernel:
    """
    Constructs the solar wind kernel with physically-meaningful initial
    parameter values expressed in scaled time units.
    """

    # We want something to capture the short scale variation, and something to
    # capture the large scale.
    short_scale_variation = RationalQuadratic(lengthscales=0.1)
    long_scale_variation = RationalQuadratic(lengthscales=1)

    trend = Linear()

    # We expect a periodic component with period roughly equal to the time it
    # takes for the same part of the sun to be subsolar to Mercury again - note
    # that this is slightly longer than a solar rotation.

    # Scale the physical period into [0,1] time units
    scaled_period = time_scaler.scale_duration(CARRINGTON_ROTATION.to(u.second).value)

    periodic_component = Periodic(
        base_kernel=SquaredExponential(),
        period=Parameter(scaled_period, trainable=False),
    )

    composite_kernel = (
        trend + short_scale_variation + long_scale_variation + periodic_component
    )

    return composite_kernel


def _interpolate_continuous_chunks(
    x: NDArray, y: NDArray, gap_threshold: float | None = None
) -> NDArray:
    """Linear interpolation that respects gaps in the data.

    Identifies continuous chunks and interpolates within each,
    rather than bridging across gaps.

    Args:
        x: Scaled time values
        y: Target values
        gap_threshold: Minimum gap size to treat as a discontinuity.
                       Defaults to 2x the median step size.
    """

    x = x.squeeze()
    y = y.squeeze()

    if gap_threshold is None:
        gaps = np.diff(x)
        gap_threshold = 2 * np.median(gaps)

    # Find indices where a new chunk begins
    split_indices = np.where(np.diff(x) > gap_threshold)[0] + 1
    x_chunks = np.split(x, split_indices)
    y_chunks = np.split(y, split_indices)

    linear_y = np.empty_like(y, dtype=float)
    start = 0
    for x_chunk, y_chunk in zip(x_chunks, y_chunks):

        end = start + len(x_chunk)

        if len(x_chunk) == 1:
            # Single point chunk — no interpolation possible, use value directly
            linear_y[start:end] = y_chunk

        else:
            linear_y[start:end] = np.interp(
                x_chunk, [x_chunk[0], x_chunk[-1]], [y_chunk[0], y_chunk[-1]]
            )

        start = end

    return linear_y


# Some useful functions to aid in the visualisation of kernels
def _plot_kernel_samples(ax: Axes, kernel: gpflow.kernels.Kernel) -> None:
    X = np.zeros((0, 1))
    Y = np.zeros((0, 1))
    model = gpflow.models.GPR((X, Y), kernel=deepcopy(kernel))
    Xplot = np.linspace(-0.6, 0.6, 100)[:, None]
    tf.random.set_seed(20220903)
    n_samples = 3
    # predict_f_samples draws n_samples examples of the function f, and returns their values at Xplot.
    fs = model.predict_f_samples(Xplot, n_samples)
    ax.plot(Xplot, fs[:, :, 0].numpy().T, label=kernel.__class__.__name__)
    ax.set_ylim(bottom=-2.0, top=2.0)
    ax.set_title("Example $f$s")


def _plot_kernel_prediction(
    ax: Axes, kernel: gpflow.kernels.Kernel, *, optimise: bool = True
) -> None:
    X = np.array([[-0.5], [0.0], [0.4], [0.5]])
    Y = np.array([[1.0], [0.0], [0.6], [0.4]])
    model = gpflow.models.GPR((X, Y), kernel=deepcopy(kernel), noise_variance=1e-3)

    if optimise:
        gpflow.set_trainable(model.likelihood, False)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

    Xplot = np.linspace(-0.6, 0.6, 100)[:, None]

    f_mean, f_var = model.predict_f(Xplot, full_cov=False)
    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)

    ax.scatter(X, Y, color="black")
    (mean_line,) = ax.plot(Xplot, f_mean, "-", label=kernel.__class__.__name__)
    color = mean_line.get_color()
    ax.plot(Xplot, f_lower, lw=0.1, color=color)
    ax.plot(Xplot, f_upper, lw=0.1, color=color)
    ax.fill_between(Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=color, alpha=0.1)
    ax.set_ylim(bottom=-1.0, top=2.0)
    ax.set_title("Example data fit")


def _plot_kernel(kernel: gpflow.kernels.Kernel, *, optimise: bool = True) -> Figure:
    fig, (samples_ax, prediction_ax) = plt.subplots(nrows=1, ncols=2)
    _plot_kernel_samples(samples_ax, kernel)
    _plot_kernel_prediction(prediction_ax, kernel, optimise=optimise)

    return fig
