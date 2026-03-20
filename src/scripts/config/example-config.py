from mvswim.modelling import GapGenerator

CONFIG = {
    "Seed": 1785,
    "Data": {
        "Spacecraft": ["Solar Orbiter"],
    },
    "Model": {
        "Inducing Points": 10,
        "Gap Generator": GapGenerator.from_normal_distributions(2 * 60, 30, 4 * 60, 30),
    },
}
