"""
Determine periodograms for solar orbiter MAG data. First for all time, then for
times 'near Mercury'.
"""

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import spiceypy as spice
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.timeseries import LombScargle
from hermpy.net import ClientSPICE
from matplotlib.axes import Axes
from sunpy.coordinates import frames
from sunpy.time import TimeRange

from mvswim.data import get_solar_orbiter_data


def main() -> None:

    time_range = TimeRange("2021-01-01", "2026-01-01")

    data: pl.DataFrame = get_solar_orbiter_data(
        time_range,
        "mag-rtn-normal-1-minute",
        quality_limit=2,
    ).drop_nans()

    # Downsample data
    data = data.group_by_dynamic("UTC", every="1h").agg(
        [pl.col(c).mean().alias(c) for c in data.columns if c != "UTC"]
    )

    # Near mercury data
    times_near_mercury = get_times_near_mercury(time_range)
    near_mercury_data = data.filter(
        pl.any_horizontal(
            *[
                pl.col("UTC").is_between(tr.start.to_datetime(), tr.end.to_datetime())
                for tr in times_near_mercury
            ]
        )
    )

    fig, axes = plt.subplots(2, 1)

    for ax, df in zip(axes, [data, near_mercury_data]):

        periodograms: Dict[str, Periodogram] = {
            "|B|": get_periodogram(df["UTC"], df["|B| [nT]"]),
            "Br": get_periodogram(df["UTC"], df["Br [nT]"]),
            "Bt": get_periodogram(df["UTC"], df["Bt [nT]"]),
            "Bn": get_periodogram(df["UTC"], df["Bn [nT]"]),
        }

        labels = ["|B|", "Br", "Bt", "Bn"]
        colours = ["black", "#D55E00", "#009E73", "#0072B2"]
        orders = [1, 0, 0, 0]

        for label, colour, order in zip(labels, colours, orders):

            # Periodograms
            periodograms[label].plot(ax, color=colour, label=label, zorder=order)

        # False alarm lines (these are all the same level so we need only
        # plot it once)
        ax.axhline(
            periodograms["|B|"].false_alarm_level(0.05),
            ls="dashed",
            lw=3,
            color="black",
            zorder=5,
            label="5% False Alarm Level",
        )

        ax.set(
            xlabel="Period [days]",
            ylabel="Power [arb.]",
        )

        ax.margins(x=0)

    axes[0].legend()

    fig.suptitle(f"Solar Orbiter\n{time_range.start} to {time_range.end}")
    axes[0].set_title("All Data")
    axes[1].set_title(
        f"Near Mercury (N={len(near_mercury_data)}, {len(near_mercury_data) / len(data) * 100:.2f}%)"
    )

    plt.tight_layout()
    plt.show()


@dataclass
class Periodogram:
    _frequencies: Sequence[float]
    _powers: Sequence[float]
    lomb_scargle: LombScargle
    unit: u.Unit

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        """
        Plots periodogram onto ax object
        """

        kwargs.setdefault("color", "black")

        ax.plot(self.periods, self.powers, *args, **kwargs)

    def plot_average(self, ax: Axes, m: int, *args, **kwargs) -> None:
        """
        Plots the periodogram convolved with a top-hat function of length m
        """

        kwargs.setdefault("color", "black")

        convolved_powers = np.convolve(self.powers, np.ones(m) / m, mode="same")

        ax.plot(self.periods, convolved_powers, *args, **kwargs)

    @property
    def frequencies(self) -> u.Quantity:
        return self._frequencies * (1 / self.unit)

    @property
    def periods(self) -> u.Quantity:
        periods = np.divide(1, self.frequencies)
        assert isinstance(periods, u.Quantity)
        return periods

    @property
    def powers(self) -> Sequence[float]:
        return self._powers

    def false_alarm_level(self, p: float = 0.05) -> float:
        return float(self.lomb_scargle.false_alarm_level(p))


def get_periodogram(time: pl.Series, var: pl.Series) -> Periodogram:

    unit: u.Unit = u.day

    assert isinstance(time.dtype, pl.Datetime)

    # Cast time to float and convert to physical unit
    time = time.cast(pl.Float64)
    time = ((time - time[0]).to_numpy() * u.ns).to(unit)  # nanoseconds to days

    lomb_scargle = LombScargle(time, var)

    min_freq = u.Quantity(1 / 60, unit**-1)
    max_freq = u.Quantity(1, unit**-1)

    frequencies, powers = lomb_scargle.autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
    )

    return Periodogram(frequencies, powers, lomb_scargle, unit=unit)


def get_times_near_mercury(
    time_range: TimeRange,
    spacecraft: str = "Solar Orbiter",
    distance_limits: Tuple[u.Quantity, u.Quantity] = (0.3075 * u.au, 0.4667 * u.au),
    latitude_limits: u.Quantity = 3.38 * u.deg,  # maximum +/-
    trajectory_resolution: u.Quantity = 1 * u.hour,
) -> Sequence[TimeRange]:
    """
    Use spiceypy and some constraints to determine times where spaecraft were
    'near Mercury'
    """

    spice_client = ClientSPICE()
    spice_client.KERNEL_LOCATIONS.update(
        {
            # Solar Orbiter
            "Solar Orbiter Frames": {
                "BASE": "http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/",
                "DIRECTORY": "kernels/fk/",
                "PATTERNS": ["solo_ANC_soc-sci-fk_V09.tf"],
            },
            "Solar Orbiter Positions": {
                "BASE": "http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/",
                "DIRECTORY": "kernels/spk/",
                "PATTERNS": [
                    "de421.bsp",
                    "solo_ANC_soc-orbit_20200210-20301118_L000_V0_00001_V01.bsp",
                ],
            },
        }
    )

    with spice_client.KernelPool():

        times: Sequence[dt.datetime] = [
            t.center.to_datetime()
            for t in time_range.split(
                int((time_range.end - time_range.start) / trajectory_resolution)
            )
        ]

        ets = spice.datetime2et(times)

        positions, _ = spice.spkpos(spacecraft, ets, "J2000", "NONE", "SUN")

    positions *= u.km

    skycoords = SkyCoord(
        CartesianRepresentation(positions.T),
        obstime=times,
        frame="icrs",
    ).transform_to(frames.HeliographicStonyhurst)

    assert isinstance(skycoords.radius, u.Quantity)
    assert isinstance(skycoords.lon, u.Quantity)
    assert isinstance(skycoords.lat, u.Quantity)

    positions_table = pl.DataFrame(
        {
            "UTC": times,
            "Radius [au]": skycoords.radius.to(u.au),
            "Longitude [deg]": skycoords.lon.to(u.deg),
            "Latitude [deg]": skycoords.lat.to(u.deg),
        }
    )

    # Filter only to times within critera
    positions_table = positions_table.filter(
        (pl.col("Radius [au]").is_between(*distance_limits))
        & (pl.col(["Latitude [deg]"]).abs() <= latitude_limits)
    )

    # We only want the time intervals for these 'near Mercury' stints.
    positions_table = positions_table.with_columns(
        (pl.col("UTC") - pl.col("UTC").shift(1)).alias("Time Step")
    )

    jump_indices = np.where(
        positions_table["Time Step"]
        > dt.timedelta(hours=trajectory_resolution.to(u.hour).value)
    )[0]

    start_indices = np.concatenate(([0], jump_indices)).tolist()
    end_indices = np.concatenate(
        (jump_indices - 1, [len(positions_table) - 1])
    ).tolist()

    time_intervals = [
        TimeRange(positions_table["UTC"][s], positions_table["UTC"][e])
        for s, e in zip(start_indices, end_indices)
    ]

    return time_intervals


if __name__ == "__main__":
    main()
