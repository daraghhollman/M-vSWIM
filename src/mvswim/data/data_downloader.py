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

DATA_PATH: Path = Path(__file__).parent.parent.parent.parent / "data/spacecraft/"

__all__ = [
    "DATA_PATH",
    "add_magnitude",
    "get_helios_data",
    "get_solar_orbiter_data",
    "get_parker_data",
    "get_messenger_data",
]


def downsample(
    df: pl.DataFrame, frequency: str, time_column: str = "UTC"
) -> pl.DataFrame:
    """

    Downsamples a polars dataframe with time column 'UTC' to an input
    frequency.

    Parameters
    ----------
    df: pl.DataFrame
        The polars dataframe to perform the operation on.

    frequency: str
        Accepted arguments are created with the following string language:
            1m (1 minute)
            1h (1 hour)
            1d (1 calendar day)

    time_column: str, {UTC}
        Which column in df corresponds to the time.
    """

    non_time_columns = [col for col in df.columns if col != time_column]

    # In polars, downsampling is a special case of the group-by operation
    df = df.group_by_dynamic(time_column, every=frequency).agg(
        [pl.col(col).mean() for col in non_time_columns]
    )

    return df


def add_magnitude(
    data: pl.DataFrame, field_columns: list[str] = ["Br [nT]", "Bt [nT]", "Bn [nT]"]
) -> pl.DataFrame:

    r, t, n = field_columns

    new_data = data.with_columns(
        (np.sqrt(pl.col(r) ** 2 + pl.col(t) ** 2 + pl.col(n) ** 2)).alias("|B| [nT]")
    )

    return new_data


def get_helios1_data(
    time_range: TimeRange,
    product: str = "40sec_mag_plasma",
    downsample_data: bool = False,
    downsample_frequency: str = "1h",
) -> pl.DataFrame:
    return get_helios_data(
        time_range,
        spacecraft=1,
        product=product,
        downsample_data=downsample_data,
        downsample_frequency=downsample_frequency,
    )


def get_helios2_data(
    time_range: TimeRange,
    product: str = "40sec_mag_plasma",
    downsample_data: bool = False,
    downsample_frequency: str = "1h",
) -> pl.DataFrame:
    return get_helios_data(
        time_range,
        spacecraft=2,
        product=product,
        downsample_data=downsample_data,
        downsample_frequency=downsample_frequency,
    )


def get_helios_data(
    time_range: TimeRange,
    spacecraft: int,
    product: str = "40sec_mag_plasma",
    downsample_data: bool = False,
    downsample_frequency: str = "1h",
):

    assert time_range.start is not None
    assert time_range.end is not None

    match product:

        case "40sec_mag_plasma":

            if spacecraft == 1:
                dataset = a.cdaweb.Dataset.helios1_40sec_mag_plasma  # type: ignore[attr-defined]

            else:
                dataset = a.cdaweb.Dataset.helios2_40sec_mag_plasma  # type: ignore[attr-defined]

            result = Fido.search(
                a.Time(time_range.start.to_string(), time_range.end.to_string()),
                dataset,
            )

            local_files = Fido.fetch(result, path=DATA_PATH)

            dataframes = []
            for path in local_files:
                cdf_file = cdflib.CDF(path)

                epoch = cdf_file.varget("Epoch")

                assert isinstance(epoch, np.ndarray)
                time = cdflib.cdfepoch.to_datetime(epoch)

                file_data = pl.DataFrame(
                    {
                        "UTC": time,
                        "Br [nT]": cdf_file.varget("B_R"),
                        "Bt [nT]": cdf_file.varget("B_T"),
                        "Bn [nT]": cdf_file.varget("B_N"),
                    }
                )

                file_data = remove_helios_nans(file_data)

                dataframes.append(file_data)

            data: pl.DataFrame = pl.concat(dataframes)

            # Downcast precision from f64 to f32 to be consistent with the
            # above spacecraft. MESSENGER's instrument recision is still
            # far below this, so there is no scientific concern.
            data = data.with_columns(
                pl.col("UTC"),
                pl.col("Br [nT]"),
                pl.col("Bt [nT]"),
                pl.col("Bn [nT]"),
            )

            # Filter data to time_range
            data = data.filter(
                (pl.col("UTC") >= time_range.start.to_datetime())
                & (pl.col("UTC") < time_range.end.to_datetime())
            )

            data = add_magnitude(data)

            # Downsample
            if downsample_data:
                data = downsample(data, frequency=downsample_frequency)

            return data

        case _:
            raise ValueError(f"Product '{product}' not yet implemented.")


def remove_helios_nans(data: pl.DataFrame) -> pl.DataFrame:

    # There are extreme negative values in this dataset which I believe to be
    # in place of missing data. These are all negative, and ~ 1e31 in
    # magnitude.
    return data.remove(pl.col("Br [nT]") < -1e30)


def get_solar_orbiter_data(
    time_range: TimeRange,
    product: str,
    quality_limit: int,
    downsample_data: bool = False,
    downsample_frequency: str = "1h",
) -> pl.DataFrame:

    assert time_range.start is not None
    assert time_range.end is not None

    match product:

        case "mag-rtn-normal-1-minute":
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
                quality = cdf_file.varget("QUALITY_FLAG")

                file_data = pl.DataFrame(
                    {
                        "UTC": time,
                        "Br [nT]": mag[:, 0],
                        "Bt [nT]": mag[:, 1],
                        "Bn [nT]": mag[:, 2],
                        "Quality": quality,
                    }
                )
                dataframes.append(file_data)

            data: pl.DataFrame = pl.concat(dataframes)

            # Filter data to time_range
            data = data.filter(
                (pl.col("UTC") >= time_range.start.to_datetime())
                & (pl.col("UTC") < time_range.end.to_datetime())
            )

            data = add_magnitude(data)

            # Filter by quality.
            # Levels 0 -> 1 are not of publication quality. Level 2 is 'survey
            # data': potentially not publication quality, but we will include
            # here.
            # https://www.cosmos.esa.int/web/soar/support-data
            length_before = len(data)
            data = data.filter(pl.col("Quality") >= quality_limit)
            length_after = len(data)

            if length_before != length_after:
                print(
                    f"Removed {length_before - length_after} timesteps with poor quality data."
                )

            # Downsample
            if downsample_data:
                data = downsample(data, frequency=downsample_frequency)

            return data

        case _:
            raise ValueError(f"Product '{product}' not yet implemented.")


def get_parker_data(
    time_range: TimeRange,
    product: str,
    downsample_data: bool = False,
    downsample_frequency: str = "1h",
) -> pl.DataFrame:

    assert time_range.start is not None
    assert time_range.end is not None

    match product.lower():

        case "psp-fld-l2-mag-rtn-1min":
            result = Fido.search(
                a.Time(time_range.start.to_string(), time_range.end.to_string()),
                a.cdaweb.Dataset.psp_fld_l2_mag_rtn_1min,  # type: ignore[attr-defined]
            )

            local_files = Fido.fetch(result, path=DATA_PATH)

            dataframes = []
            for path in local_files:
                cdf_file = cdflib.CDF(path)

                epoch = cdf_file.varget("epoch_mag_RTN_1min")

                epoch_quality_flags = cdf_file.varget("epoch_quality_flags")
                quality_flags = cdf_file.varget("psp_fld_l2_quality_flags")

                assert isinstance(epoch, np.ndarray)
                assert isinstance(epoch_quality_flags, np.ndarray)

                time = cdflib.cdfepoch.to_datetime(epoch)
                time_quality = cdflib.cdfepoch.to_datetime(epoch_quality_flags)

                mag = np.array(cdf_file.varget("psp_fld_l2_mag_RTN_1min"))

                file_data = pl.DataFrame(
                    {
                        "UTC": time,
                        "Br [nT]": mag[:, 0],
                        "Bt [nT]": mag[:, 1],
                        "Bn [nT]": mag[:, 2],
                    }
                )

                # Quality flags don't always line up with the data. We must do
                # a join on nearest times.
                quality_df = pl.DataFrame(
                    {
                        "UTC_quality": time_quality,
                        "Quality": quality_flags,
                    }
                )
                # Join on nearest timestamp to handle off-by-one misalignments.
                # This correctly handles cases where quality flag timestamps differ slightly.
                file_data = file_data.join_asof(
                    quality_df.sort("UTC_quality"),
                    left_on="UTC",
                    right_on="UTC_quality",
                    strategy="nearest",
                )

                # Remove columns
                file_data = file_data.drop(["UTC_quality"])

                dataframes.append(file_data)

            data: pl.DataFrame = pl.concat(dataframes)

            # Filter data to time_range
            data = data.filter(
                (pl.col("UTC") >= time_range.start.to_datetime())
                & (pl.col("UTC") < time_range.end.to_datetime())
            )

            data = add_magnitude(data)

            # Exclude anything with flags for bad quality.
            # https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/psp_fld_l2_mag_rtn_1min_00000000_v01.skt
            length_before = len(data)
            data = data.filter(pl.col("Quality") == 0)
            length_after = len(data)

            if length_before != length_after:
                print(
                    f"Removed {length_before - length_after} timesteps with poor quality data."
                )

            # Downsample
            if downsample_data:
                data = downsample(data, frequency=downsample_frequency)

            return data

        case _:
            raise ValueError(f"Product '{product}' not yet implemented.")


def get_messenger_data(
    time_range: TimeRange,
    product: str,
    downsample_data: bool = False,
    downsample_frequency: str = "1h",
) -> pl.DataFrame:
    assert time_range.start is not None
    assert time_range.end is not None

    match product:

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
                    pl.col("Br [nT]"),
                    pl.col("Bt [nT]"),
                    pl.col("Bn [nT]"),
                )

                dataframes.append(file_data)

            data = pl.concat(dataframes)

            if downsample_data:
                data = downsample(data, frequency=downsample_frequency)

            return data

        case _:
            raise ValueError(f"Product '{product}' not yet implemented.")


def get_bepicolombo_data(time_range: TimeRange, instrument: str = "MPO-MAG"):
    pass
