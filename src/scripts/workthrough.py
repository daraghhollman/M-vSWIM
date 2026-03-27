"""
Just a script to work through the required components for modelling in this work.
"""

import datetime as dt
from pathlib import Path

import tensorflow as tf
from numpy.typing import NDArray
from sunpy.time import TimeRange

from mvswim.data import MAGData, get_solar_orbiter_data
from mvswim.modelling import GapGenerator, SolarWindModel, gap_generator

# Create a unique log directory for this run
LOG_DIR = Path("./logs/" + dt.datetime.now().strftime("%Y-%m-%d--%H:%M"))

SEED = 1785
tf.random.set_seed(SEED)

# To train a model, we need an interval of data within which we create artificial gaps

# Make Data object
mag: MAGData = MAGData(
    get_solar_orbiter_data(
        TimeRange("2021-01-01", "2021-01-02"),
        product="mag-rtn-normal-1-minute",
        quality_limit=2,
    ),
    metadata={"Spacecraft": "Solar Orbiter"},
)

X: NDArray = mag.data["UTC"].to_numpy().reshape(-1, 1)
Y: NDArray = mag.data["|B| [nT]"].to_numpy().reshape(-1, 1).astype("float64")

# gap_generator = GapGenerator.from_constants(
#     gap_size=2 * 60, gap_interval=4 * 60, seed=SEED
# )
gap_generator = GapGenerator.from_normal_distributions(2 * 60, 30, 4 * 60, 30)
X, Y = gap_generator.create_gaps(X, Y)

# For now we include the full dataset. This model class has functionality to
# separate a training and testing set.
model = SolarWindModel.build(
    input=X,
    output=Y,
    n_inducing_points=50,
    seed=SEED,
    log_directory=LOG_DIR,
)

model.train_model()

model.quicklook()
