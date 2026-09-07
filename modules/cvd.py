import numpy as np


def simulate_cvd(rgb, deficiency):
    """
    Simulate how an RGB color may appear
    under a specific type of color vision deficiency.
    """

    rgb = np.array(rgb, dtype=float) / 255.0

    if deficiency == "protanopia":
        matrix = np.array([
            [0.567, 0.433, 0.000],
            [0.558, 0.442, 0.000],
            [0.000, 0.242, 0.758]
        ])

    elif deficiency == "deuteranopia":
        matrix = np.array([
            [0.625, 0.375, 0.000],
            [0.700, 0.300, 0.000],
            [0.000, 0.300, 0.700]
        ])

    elif deficiency == "tritanopia":
        matrix = np.array([
            [0.950, 0.050, 0.000],
            [0.000, 0.433, 0.567],
            [0.000, 0.475, 0.525]
        ])

    else:
        raise ValueError("Unknown CVD type")

    simulated = np.dot(rgb, matrix.T)

    simulated = np.clip(simulated, 0, 1)

    return tuple((simulated * 255).astype(int))

