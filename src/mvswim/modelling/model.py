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
import psutil
import pynvml
import tensorflow as tf
import tensorflow.summary
from gpflow import Parameter
from gpflow.kernels import Kernel
from gpflow.models import SGPR
from gpflow.monitor import (
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
from gpflow.optimizers import Scipy
from keras.optimizers import Optimizer
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from tensorflow import Tensor

from mvswim.scalling import KernelScaler, TimeScaler

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
        kernel: Kernel,
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

        # Use K-Means Clustering to determine good inducing points
        kmeans = MiniBatchKMeans(
            n_clusters=n_inducing_points, random_state=seed, n_init="auto"
        )
        kmeans.fit(X)
        inducing_points = kmeans.cluster_centers_

        kernel_scaler = KernelScaler(time_scaler)
        scaled_kernel = kernel_scaler.scale(kernel)

        gpmodel = SGPR((X, Y), kernel=scaled_kernel, inducing_variable=inducing_points)

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

    def train_model(self, log_gpu: bool = False) -> None:
        """
        Perform iterations of training.
        """

        training_log_dir = str(
            self.log_directory / f"{dt.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        )

        monitors = []

        # Log system metrics
        cpu_monitor = ScalarToTensorBoard(
            training_log_dir,
            lambda: get_system_metrics()["cpu-usage"],
            "system/CPU Usage",
        )
        monitors.append(cpu_monitor)

        memory_monitor = ScalarToTensorBoard(
            training_log_dir,
            lambda: get_system_metrics()["ram-usage"],
            "system/RAM Usage",
        )
        monitors.append(memory_monitor)

        if log_gpu:
            vram_monitor = ScalarToTensorBoard(
                training_log_dir,
                lambda: get_gpu_metrics()["vram-usage"],
                "system/VRAM Usage",
            )
            power_monitor = ScalarToTensorBoard(
                training_log_dir,
                lambda: get_gpu_metrics()["power-usage"],
                "system/Power Consumption [W]",
            )
            gpu_monitor = ScalarToTensorBoard(
                training_log_dir,
                lambda: get_gpu_metrics()["gpu-usage"],
                "system/GPU Usage",
            )
            monitors.extend([gpu_monitor, vram_monitor, power_monitor])

        # Log training loss
        loss_monitor = ScalarToTensorBoard(
            training_log_dir,
            self.get_training_loss,
            "training/Loss",
        )
        monitors.append(loss_monitor)

        # Log model
        model_monitor = ModelToTensorBoard(
            training_log_dir,
            self.model,
        )
        monitors.append(model_monitor)

        task_group = MonitorTaskGroup(monitors, period=3)
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

        # Check if input contains nan
        if np.isnan(testing_y).any():
            print(testing_y)
            print(len(np.isnan(testing_y)))

        self.log(f"TESTING")
        # We need to scale our inputs
        testing_x = self.time_scaler.time_to_numeric(testing_x)

        # Get model predictions for this time range
        y_mean, _ = self.model.predict_y(testing_x)

        # If the model couldn't fit the predict function will produce nan
        # values. Usually this means not enough inducing points were used.
        # In these cases, we return nan for the metrics and continue.
        return_nans: bool = False
        if np.isnan(y_mean).any():
            self.log("Model failed to predict, returning nans.")
            return_nans = True

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

            if return_nans:
                model_metrics.update({label: np.nan})
                linear_metrics.update({label: np.nan})

                continue

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


def get_system_metrics():
    return {
        "cpu-usage": psutil.cpu_percent(),
        "ram-usage": psutil.virtual_memory().percent,
    }


def get_gpu_metrics():

    # Fetch the first gpu
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return {
        "gpu-usage": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,  # %
        "power-usage": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,  # W
        "vram-usage": memory.used,
    }
