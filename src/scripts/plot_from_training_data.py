import sys
from pathlib import Path

import matplotlib.pyplot as plt

from mvswim.modelling import plot_from_training_data

path = Path(sys.argv[1])
plot_from_training_data(path)
plt.show()
