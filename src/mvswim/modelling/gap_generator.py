from typing import Any, Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["GapGenerator"]


class GapGenerator:

    def __init__(
        self,
        get_gap_size: Callable,
        get_gap_interval: Callable,
        seed: int | None = None,
    ) -> None:

        self.get_gap_size = get_gap_size
        self.get_gap_interval = get_gap_interval
        self.rng = np.random.default_rng(seed)

    def generate_gaps(self, n) -> List[Tuple[int, int]]:
        """
        Returns list of (start, end) indices for gaps within length n.
        """
        gaps = []
        start_index = 0
        while start_index < n:

            # We should always have data at the start, so we first move by the
            # interval, then create the gap.
            interval = self.get_gap_interval(self.rng)
            start_index += interval

            # Similarly, we always want an interval sized gap at the end.
            if start_index >= n - interval:
                break

            size = self.get_gap_size(self.rng)

            # We use min as a safety check here, so we don't try to make gaps
            # longer than the length of data left. However, this should be
            # caught by the above (n - interval) check for all cases where
            # intervals are longer than gaps.
            end_index = min(n, start_index + size)

            # Record this gap
            gaps.append((start_index, end_index))

            start_index = end_index

        return gaps

    def generate_mask(self, n_samples):
        """
        Returns boolean mask of shape (n_samples,)
        True  = keep
        False = gap
        """
        mask = np.ones(n_samples, dtype=bool)
        gaps = self.generate_gaps(n_samples)

        for start, end in gaps:
            mask[start:end] = False

        return mask

    def create_gaps(self, *arrays: NDArray[Any]) -> Tuple[NDArray, ...]:
        """
        Creates the gaps in the input numpy array according the generator
        constructed with. Applies aligned gaps to multiple arrays.
        Returns tuple of masked arrays.
        """

        if len(arrays) == 0:
            raise ValueError("At least one array required")

        n_samples = arrays[0].shape[0]

        # Validate alignment
        for array in arrays:
            if array.shape[0] != n_samples:
                raise ValueError("All arrays must share first dimension")

        mask = self.generate_mask(n_samples)

        return tuple(array[mask] for array in arrays)

    @classmethod
    def from_constants(
        cls, gap_size: float, gap_interval: float, seed: int | None = None
    ):
        return cls(
            get_gap_size=lambda _: gap_size,
            get_gap_interval=lambda _: gap_interval,
            seed=seed,
        )
