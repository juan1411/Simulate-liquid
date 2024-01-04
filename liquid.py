from pygame.math import Vector2
from pygame.draw import circle
from numpy import pi
import numpy as np
from numba import njit

from constants import *
    

@njit(cache=True)
def smoothing_kernel(dst: float | np.ndarray) -> float | np.ndarray:
    value = SMOOTHING_RADIUS - dst
    value = np.clip(value, a_min=0, a_max=None)

    return (value ** 3) / VOLUME

@njit(cache=True)
def smoothing_kernel_derivative(dst: float | np.ndarray) -> float | np.ndarray:
    value = SMOOTHING_RADIUS - dst
    value = np.clip(value, a_min=0, a_max=None)

    return -3 * (value ** 2) / VOLUME


def create_particules(num_particules:int = NUM_PARTICULES, mode:str = "random") -> list[np.ndarray]:
    """Function to create the positions of the particules.
    
    `Mode` options:
        - `random` for random positions inside of the tank
        - `grid` for equal spacing particules
    """
    spacing = round(RADIUS * 2.5, 0)
    per_row = int(200/RADIUS)
    n_rows = num_particules//per_row + (1 if num_particules%per_row > 0 else 0)
    inicial_x = round(CENTER_TANK.x - per_row * spacing / 2, 0)
    inicial_y = round(CENTER_TANK.y - n_rows * spacing / 2, 0)

    positions = []
    for i in range(num_particules):

        if mode == "random":
            # TODO: random positions INSIDE the tank
            pos = np.random.randint(WIN_RES * 0.1, WIN_RES * 0.9, 2)

        elif mode == "grid":
            pos = (inicial_x + (i%per_row - 1) * spacing, inicial_y + (i//per_row + 1) * spacing)
            pos = np.array(pos)

        positions.append(pos)

    return positions
    

def calculate_density_old(positions: np.ndarray, ref: np.ndarray) -> float:
    influence = 0
    for p in positions:
        dst = np.sqrt(np.sum( (p - ref)**2, axis=-1))
        influence += smoothing_kernel(dst)
    
    return round(influence[0] * MASS, 6) * SCALING_FACTOR_DENSITY


@njit(cache=True)
def calculate_density(positions: np.ndarray, ref: np.ndarray) -> float:
    dst = np.sqrt(np.sum((positions - ref)**2, axis=-1))
    # print(dst.shape)

    influence = np.sum(smoothing_kernel(dst))
    return round(influence * MASS, 6) * SCALING_FACTOR_DENSITY
    
# @njit
def calculate_pressure_force(positions: np.ndarray, densities: np.ndarray, ref: np.ndarray) -> np.ndarray:
    assert positions.shape[0] == densities.shape[0]

    dir = (positions - ref)
    dst = np.sqrt(np.sum(dir**2, axis=-1))

    slope = smoothing_kernel_derivative(dst)
    pressure = density_to_pressure(densities)

    div = dst * densities
    div = np.where(div > 0, div, -1)
    multiplier = pressure * slope * MASS * SCALING_FACTOR_DENSITY / div

    influences = dir.ravel('F') * np.concatenate([multiplier, multiplier], axis=0)
    influences = influences.reshape( (len(influences)//2, 2), order='F')
    influences = np.clip(influences, a_min=0, a_max=None)

    return np.sum(influences, axis=0)


@njit(cache=True)
def density_to_pressure(density: float | np.ndarray) -> float | np.ndarray:
    return (TARGET_DENSITY - density) * PRESSURE_FACTOR