import numpy as np
from numba import njit

from constants import *


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
def smoothing_kernel(dst: float | np.ndarray) -> float | np.ndarray:
    value = SMOOTHING_RADIUS - dst
    value = np.clip(value, a_min=0, a_max=None)

    return (value ** 2) / VOLUME

@njit(cache=True)
def smoothing_kernel_derivative(dst: float | np.ndarray) -> float | np.ndarray:
    value = dst - SMOOTHING_RADIUS
    value = np.clip(value, a_min=None, a_max=0)

    return 2 * value * FACTOR_SLOPE / VOLUME


@njit(cache=True)
def calculate_density(positions: np.ndarray, ref: np.ndarray) -> float:
    dst = np.sqrt(np.sum((positions - ref)**2, axis=-1))

    influence = np.sum(smoothing_kernel(dst))
    return influence * MASS * FACTOR_DENSITY
    
@njit(cache=True)
def calculate_pressure_force(
    positions: np.ndarray, densities: np.ndarray,
    ref_pos: np.ndarray, ref_dens: float
) -> np.ndarray:
    assert positions.shape[0] == densities.shape[0]

    dir = (positions - ref_pos)
    dst = np.sqrt(np.sum(dir**2, axis=-1))

    slope = smoothing_kernel_derivative(dst)
    shared_pressure = (density_to_pressure(densities) + density_to_pressure(ref_dens)) / 2

    div = dst * densities
    div = np.where(div > 0, div, np.random.randint(5, 10, div.shape)) # NOTE: zero divison Error
    multiplier = shared_pressure * slope * MASS / div

    influences = dir.copy()
    influences[:, 0] = dir[:, 0] * multiplier
    influences[:, 1] = dir[:, 1] * multiplier

    # NOTE: help to debugging
    # ind = [0, 1, 2, 30, 31, 32, 60, 61, 62]
    # print('Dirs:', dir[ind])
    # print("Dists:", dst[ind])
    # print("Dens:", densities[ind])
    # print("Slope:", slope[ind])
    # print("Pres:", shared_pressure[ind])
    # print("Res", influences[ind])

    return np.sum(influences, axis=0)


@njit(cache=True)
def density_to_pressure(density: float | np.ndarray) -> float | np.ndarray:
    return (density - TARGET_DENSITY) * FACTOR_PRESSURE