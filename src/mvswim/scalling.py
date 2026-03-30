from gpflow.kernels import Kernel
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler


class TimeScaler:

    def __init__(self, fit_data: NDArray) -> None:

        self._fit_data: NDArray = fit_data
        self._scaler: MinMaxScaler = self.init_transform()
        self._data_range: float = self._scaler.data_range_[0]

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

    def scale_duration(self, duration: float) -> float:
        """
        Converts a physical duration (in units of data cadence) into the scaled
        [0, 1] time units used by the model. Durations scale by the data range
        only — no offset is applied, unlike point values.
        """
        return duration / self._data_range


class KernelScaler:
    """
    Rescales kernel lengthscales from physical durations (seconds) into the
    [0, 1] space defined by a fitted TimeScaler.

    Using this class means that Kernels can be defined in physically meaningful
    units.

    Parameters
    ----------
    time_scaler : TimeScaler
        A TimeScaler instance that has already been fitted to the training data.

    Example
    -------
    scaler = TimeScaler(X_train)
    kernel_scaler = KernelScaler(scaler)

    k = gpflow.kernels.SquaredExponential(lengthscales=30 * 86400)  # 30 days in seconds
    kernel_scaler.scale(k)
    """

    def __init__(self, time_scaler: "TimeScaler") -> None:
        self._time_scaler = time_scaler

    def scale(self, kernel: Kernel) -> Kernel:
        """
        Rescale all lengthscales (and periods) in the kernel tree in-place.
        Returns the kernel for convenient chaining.
        """
        self._scale_recursive(kernel)
        return kernel

    def _scale_recursive(self, kernel: Kernel) -> None:
        """Walk composite kernels (Sum, Product) recursively."""
        if hasattr(kernel, "kernels"):
            for child in kernel.kernels:
                self._scale_recursive(child)
            return

        # Scale any lengthscales
        if hasattr(kernel, "lengthscales"):
            self._rescale_param(kernel, "lengthscales")

        # Scale any periods
        if hasattr(kernel, "period"):
            self._rescale_param(kernel, "period")

    def _rescale_param(self, kernel: Kernel, param: str) -> None:
        raw_seconds = getattr(kernel, param).numpy()
        scaled = self._time_scaler.scale_duration(float(raw_seconds))
        getattr(kernel, param).assign(scaled)
