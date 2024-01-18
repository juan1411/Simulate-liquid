import numpy as np
from numba import njit, jit

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
            pos = np.array(pos).reshape((1, 2))

        positions[i] = pos

    return positions


@njit
def smoothing_kernel(dst: float | np.ndarray) -> float | np.ndarray:
    """Min. value: 0
    Max. value: 6 * R^2 / pi * R^4
    """
    value = (SMOOTHING_RADIUS - dst) / PIX_TO_UN
    value = np.clip(value, a_min=0, a_max=None)

    return (value ** 2) / VOLUME

@njit
def smoothing_kernel_derivative(dst: float | np.ndarray) -> float | np.ndarray:
    """Min. value: (-12) * R (* FACTOR_SLOPE) / pi * R^4
    Max. value: 0
    """
    value = (dst - SMOOTHING_RADIUS) / PIX_TO_UN
    value = np.clip(value, a_min=SMOOTHING_RADIUS/-PIX_TO_UN, a_max=0)

    return 2 * value * FACTOR_SLOPE / VOLUME


@njit
def calculate_density(positions: np.ndarray, ref: np.ndarray) -> float:
    """Min. value: 0
    Max. value: MAX_DENSITY (approx. XX times the density of one particule)
    ---
    'XX' is a function of SMOOTHING_RADIUS; how many particules can be inside of it?
    """
    dst = np.sqrt(np.sum((positions - ref)**2, axis=-1))

    influence = np.sum(smoothing_kernel(dst))
    return influence * MASS * FACTOR_DENSITY


@njit
def density_to_pressure(density: float | np.ndarray, factor: float = FACTOR_PRESSURE) -> float | np.ndarray:
    """Min. value: -TARGET_DENSITY * FACTOR_PRESSURE
    Max. value: approx. (MAX_DENSITY -TARGET_DENSITY) * FACTOR_PRESSURE
    """
    return np.abs(density - TARGET_DENSITY) * factor


@njit
def calculate_pressure_force(
    positions: np.ndarray, densities: np.ndarray,
    ref_pos: np.ndarray, ref_dens: float,
    factor: float = FACTOR_PRESSURE
) -> np.ndarray:
    assert positions.shape[0] == densities.shape[0]

    dir = (positions - ref_pos)
    dst = np.sqrt(np.sum(dir**2, axis=-1))

    div = dst * densities
    # NOTE: np.where for zero divison error
    div = np.where(div > 0, div, div+0.1)
    
    slope = smoothing_kernel_derivative(dst)

    # gradient with pressure
    shared_pressure = (density_to_pressure(densities, factor) + density_to_pressure(ref_dens, factor)) / 2
    multiplier = shared_pressure * slope / div

    influences = dir.copy()
    influences[:, 0] = dir[:, 0] * multiplier
    influences[:, 1] = dir[:, 1] * multiplier

    # # NOTE: help to debugging
    # ind = [0, 1, 2, 30, 31, 32, 60, 61, 62]
    # print('Dirs:', dir[ind])
    # print("Dists:", dst[ind])
    # print("Dens:", densities[ind])
    # print("Slope:", slope[ind])
    # print("Pres:", shared_pressure[ind])
    # print("Res", influences[ind])

    return np.sum(influences, axis=0) * MASS * 1000

@njit
def exemple_func(pos: np.ndarray) -> np.ndarray:
    """Function to test the gradient
    Min. value: -1
    Max. value: +1
    """
    assert len(pos.shape) == 2

    return np.cos((pos[:, 1] / PIX_TO_UN) -3 + np.sin(pos[:, 0] / PIX_TO_UN))

@njit
def calculate_exemple(positions: np.ndarray, densities:np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Function to test the gradient
    Min. value: -1
    Max. value: +1
    """
    dst = np.sqrt(np.sum((positions - ref)**2, axis=-1))

    property = exemple_func(positions)
    influence = np.sum(smoothing_kernel(dst) * property / densities)

    # to range (-1, +1):
    influence *= np.pi * (SMOOTHING_RADIUS**2) / 6
    return influence * MASS


@njit
def calculate_exemple_gradient(
    positions: np.ndarray, densities: np.ndarray,
    ref_pos: np.ndarray
) -> np.ndarray:
    assert positions.shape[0] == densities.shape[0]

    dir = (positions - ref_pos)
    dst = np.sqrt(np.sum(dir**2, axis=-1))

    div = dst * densities
    # NOTE: np.where for zero divison error
    div = np.where(div > 0, div, div+0.1)
    
    slope = smoothing_kernel_derivative(dst)

    # testing gradient
    property = exemple_func(ref_pos)
    multiplier = property * slope / div

    influences = dir.copy()
    influences[:, 0] = dir[:, 0] * multiplier
    influences[:, 1] = dir[:, 1] * multiplier

    return np.sum(influences, axis=0) * MASS * 20_000

@njit
def calculate_mouse_force(
    mouse_pos:np.ndarray, positions:np.ndarray,
    vels:np.ndarray, rad:float, strength:float
) -> np.ndarray:
    force = np.zeros(positions.shape)
    offset = mouse_pos - positions
    dst = np.sqrt(np.sum(offset**2, axis=-1))
    dst = np.stack((dst, dst), axis=1)
    center_t = 1 - dst/rad
    
    dir_to_input = np.where(dst > 0.0001, offset/dst, force)
    force += np.where(dst < rad, (dir_to_input *strength -vels)*center_t, force)

    return force