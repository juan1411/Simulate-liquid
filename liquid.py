from pygame.math import Vector2
from pygame.draw import circle
from numpy import pi
import numpy as np
from numba import njit

from constants import *

class particule:

    def __init__(self, pos: tuple | list | np.ndarray | Vector2, rad: float = RADIUS):
        self.pos = pos if isinstance(pos, Vector2) else Vector2(pos[0], pos[1])
        self.rad = rad
        self.vel: Vector2 = Vector2(0, 0)
        self.density: float = None
        self.pressure: Vector2 = Vector2(0, 0)
        self.color = COLOR_WATER

    def get_pos(self) -> np.ndarray:
        return np.array(self.pos[:]).reshape((1, 2))

    def draw(self, screen):
        circle(screen, self.color, self.pos, self.rad)
    

@njit
def smoothing_kernel(dst: float | np.ndarray) -> float | np.ndarray:
    value = SMOOTHING_RADIUS - dst
    value = np.clip(value, a_min=0, a_max=None)

    return (value ** 3) / VOLUME

@njit
def smoothing_kernel_derivative(dst: float | np.ndarray) -> float | np.ndarray:
    value = SMOOTHING_RADIUS - dst
    value = np.clip(value, a_min=0, a_max=None)

    return -3 * (value ** 2) / VOLUME


def create_particules(num_particules: int = NUM_PARTICULES) -> list[particule]:
    spacing = int(RADIUS * 1.5)
    per_row = 130
    particules = []
    for i in range(num_particules):
        pos = np.random.randint(WIN_RES * 0.1, WIN_RES * 0.9, 2)
        # pos = (100 + (i%per_row - 1) * spacing, 130 + (i//per_row + 1) * spacing)
        particules.append(particule(pos))

    return particules
    

def calculate_density_old(positions: np.ndarray, ref: np.ndarray) -> float:
    influence = 0
    for p in positions:
        dst = np.sqrt(np.sum( (p - ref)**2, axis=-1))
        influence += smoothing_kernel(dst)
    
    return round(influence[0] * MASS, 6) * SCALING_FACTOR_DENSITY


@njit
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


@njit
def density_to_pressure(density: float | np.ndarray) -> float | np.ndarray:
    return (TARGET_DENSITY - density) * PRESSURE_FACTOR