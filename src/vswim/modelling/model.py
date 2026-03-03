from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

__all__ = ["TimeScaler"]


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
