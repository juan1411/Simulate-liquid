import numpy as np
from numba import njit

from constants import *
    

@njit(cache=True)
def smoothing_kernel(dst: float | np.ndarray) -> float | np.ndarray:
    value = SMOOTHING_RADIUS - dst
    value = np.clip(value, a_min=0, a_max=None)

    return (value ** 2) / VOLUME

@njit(cache=True)
def smoothing_kernel_derivative(dst: float | np.ndarray) -> float | np.ndarray:
    value = dst - SMOOTHING_RADIUS
    value = np.clip(value, a_min=0, a_max=None)

    return value / VOLUME


def create_particules(num_particules:int = NUM_PARTICULES, mode:str = "random") -> np.ndarray:
    """Function to create the positions of the particules.
    
    `Mode` options:
        - `random` for random positions inside of the tank
        - `grid` for equal spacing particules
    """
    spacing = round(RADIUS * 2.5, 0)
    per_row = int(300/RADIUS)
    n_rows = num_particules//per_row + (1 if num_particules%per_row > 0 else 0)
    inicial_x = round(CENTER_TANK.x - (per_row -1) * spacing / 2, 0)
    inicial_y = round(CENTER_TANK.y - n_rows * spacing / 2, 0)

    positions = np.zeros((num_particules, 2))
    for i in range(num_particules):

        if mode == "random":
            pos = np.random.randint(
                (TANK[0] +10, TANK[1] +10),
                (TANK[0] +TANK[2] -20, TANK[1] +TANK[3] -20),
                2
            )

        elif mode == "grid":
            pos = (inicial_x + (i%per_row) * spacing, inicial_y + (i//per_row +1) * spacing)
            pos = np.array(pos)

        positions[i] = pos

    return positions

@njit(cache=True)
def calculate_density(positions: np.ndarray, ref: np.ndarray) -> float:
    dst = np.sqrt(np.sum((positions - ref)**2, axis=-1))

    influence = np.sum(smoothing_kernel(dst))
    return round(influence * MASS, 6) * SCALING_FACTOR_DENSITY
    
@njit(cache=True)
def calculate_pressure_force(positions: np.ndarray, densities: np.ndarray, ref: np.ndarray) -> np.ndarray:
    assert positions.shape[0] == densities.shape[0]

    dir = (positions - ref)
    dst = np.sqrt(np.sum(dir**2, axis=-1))

    slope = smoothing_kernel_derivative(dst)
    pressures = density_to_pressure(densities)

    div = dst * densities
    div = np.where(div > 0, div, -1) # NOTE: zero divison Error, -1 is valid?
    multiplier = pressures * slope * MASS / div

    influences = dir.copy()
    influences[:, 0] = dir[:, 0] * multiplier
    influences[:, 1] = dir[:, 1] * multiplier

    return np.sum(influences, axis=0)


@njit(cache=True)
def density_to_pressure(density: float | np.ndarray) -> float | np.ndarray:
    return (density - TARGET_DENSITY) * PRESSURE_FACTOR