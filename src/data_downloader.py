from pathlib import Path

import cdflib
import numpy as np
import polars as pl
import sunpy_soar  # Required to register the Solar Orbiter Fido clients. Will throw static typing and linting issues :/
from astropy.io import ascii
from astropy.table import Table
from astropy.time import Time
from hermpy.net import ClientMESSENGER
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.time import TimeRange

DATA_PATH: Path = Path(__file__).parent.parent / "data/solo/"


def main():
    # A quick example of how to use each of the below functions to access data.

    data = get_solar_orbiter_data(TimeRange("2021-01-01 00:00", "2021-01-02 00:00"))

    print(data)

    data = get_parker_data(TimeRange("2021-01-01 00:00", "2021-01-02 00:00"))

    print(data)

    data = get_messenger_data(TimeRange("2012-01-01 00:00", "2012-01-02 00:00"))

    print(data)


def get_solar_orbiter_data(time_range: TimeRange, instrument: str = "MAG"):

    assert time_range.start is not None
    assert time_range.end is not None

    match instrument:

        case "MAG":
            result = Fido.search(
                a.Time(time_range.start.to_string(), time_range.end.to_string()),
                a.soar.Product("mag-rtn-normal-1-minute"),  # type: ignore[attr-defined]
            )

            local_files = Fido.fetch(result, path=DATA_PATH)

            dataframes = []
            for path in local_files:
                cdf_file = cdflib.CDF(path)

                epoch = cdf_file.varget("EPOCH")

                assert isinstance(epoch, np.ndarray)
                time = cdflib.cdfepoch.to_datetime(epoch)

                mag = np.array(cdf_file.varget("B_RTN"))

                file_data = pl.DataFrame(
                    {
                        "UTC": time,
                        "Br [nT]": mag[:, 0],
                        "Bt [nT]": mag[:, 0],
                        "Bn [nT]": mag[:, 0],
                    }
                )
                dataframes.append(file_data)

            return pl.concat(dataframes)

        case _:
            raise ValueError(f"Instrument '{instrument}' not yet implemented.")


def get_parker_data(time_range: TimeRange, instrument: str = "MAG"):

    assert time_range.start is not None
    assert time_range.end is not None

    match instrument:

        case "MAG":
            result = Fido.search(
                a.Time(time_range.start.to_string(), time_range.end.to_string()),
                a.cdaweb.Dataset.psp_fld_l2_mag_rtn_1min,  # type: ignore[attr-defined]
            )

            local_files = Fido.fetch(result, path=DATA_PATH)

            dataframes = []
            for path in local_files:
                cdf_file = cdflib.CDF(path)

                epoch = cdf_file.varget("epoch_mag_RTN_1min")

                assert isinstance(epoch, np.ndarray)
                time = cdflib.cdfepoch.to_datetime(epoch)

                mag = np.array(cdf_file.varget("psp_fld_l2_mag_RTN_1min"))

                file_data = pl.DataFrame(
                    {
                        "UTC": time,
                        "Br [nT]": mag[:, 0],
                        "Bt [nT]": mag[:, 0],
                        "Bn [nT]": mag[:, 0],
                    }
                )

                dataframes.append(file_data)

            return pl.concat(dataframes)

        case _:
            raise ValueError(f"Instrument '{instrument}' not yet implemented.")


def get_messenger_data(time_range: TimeRange, instrument: str = "MAG"):
    assert time_range.start is not None
    assert time_range.end is not None

    match instrument:

        case "MAG":
            client = ClientMESSENGER()

            client.query(time_range, "MAG RTN 60s")
            local_files = client.fetch()

            dataframes = []
            for path in local_files:

                table = ascii.read(path)
                assert type(table) == Table

                # Extract time information
                year = table.columns[0]
                doy = table.columns[1]
                hour = table.columns[2]
                minute = table.columns[3]
                second = table.columns[4]

                yday = [
                    f"{y}:{d:03d}:{h:02d}:{m:02d}:{s}"
                    for y, d, h, m, s in zip(year, doy, hour, minute, second)
                ]
                time = (
                    Time(yday, format="yday", scale="utc")
                    .to_datetime()
                    .astype("datetime64[ns]")
                )

                # Averaged Product
                # Note: the time column for averaged data products is at the centre
                # of the averaging window.
                file_data = pl.DataFrame(
                    {
                        "UTC": time,
                        "Br [nT]": table.columns[10],
                        "Bt [nT]": table.columns[11],
                        "Bn [nT]": table.columns[12],
                    },
                )

                # Downcast precision from f64 to f32 to be consistent with the
                # above spacecraft. MESSENGER's instrument recision is still
                # far below this, so there is no scientific concern.
                file_data = file_data.with_columns(
                    pl.col("UTC"),
                    pl.col("Br [nT]").cast(pl.Float32),
                    pl.col("Bt [nT]").cast(pl.Float32),
                    pl.col("Bn [nT]").cast(pl.Float32),
                )

                dataframes.append(file_data)

            return pl.concat(dataframes)

        case _:
            raise ValueError(f"Instrument '{instrument}' not yet implemented.")


def get_bepicolombo_data(time_range: TimeRange, instrument: str = "MPO-MAG"):
    pass


if __name__ == "__main__":
    main()
