from pygame.math import Vector2
from pygame.draw import circle
from numpy import pi
import numpy as np
from numba import njit

from constants import *

class particule:

    def __init__(self, pos: tuple | list | np.ndarray, rad: float = RADIUS):
        self.pos = pos if isinstance(pos, np.ndarray) else np.array(pos)
        self.rad = rad
        self.vel: Vector2 = Vector2(0, 0)
        self.density: float = None
        self.pressure: Vector2 = Vector2(0, 0)
        self.color = COLOR_WATER

        # valid pos:
        self.pos = self.pos.reshape((1, 2))

    def draw(self, screen):
        circle(screen, self.color, self.pos, self.rad)
    

@njit
def smoothing_kernel(dst: float | np.ndarray) -> float | np.ndarray:
    value = SMOOTHING_RADIUS - dst
    value = np.clip(value, a_min=0, a_max=None)

    return (value ** 3) / VOLUME

def smoothing_kernel_derivative(dst: float | np.ndarray) -> float | np.ndarray:
    value = SMOOTHING_RADIUS - dst
    value = np.clip(value, a_min=0, a_max=None)

    return -3 * (value ** 2) / VOLUME


def create_particules(num_particules: int = NUM_PARTICULES) -> list[particule]:
    spacing = 7
    per_row = 130
    particules = []
    for i in range(num_particules):
        # pos = np.random.randint(WIN_RES * 0.1, WIN_RES * 0.9, 2)
        pos = (100 + (i%per_row - 1) * spacing, 130 + (i//per_row + 1) * spacing)
        particules.append(particule(pos))

    return particules
    

def calculate_density_old(positions: np.ndarray, ref: np.ndarray) -> float:
    ref = ref.reshape((1, 2))

    influence = 0
    for p in positions:
        dst = np.sqrt(np.sum( (p - ref)**2, axis=-1))
        influence += smoothing_kernel(dst)
    
    return round(influence[0] * MASS, 6) * SCALING_FACTOR_DENSITY


@njit
def calculate_density(positions: np.ndarray, ref: np.ndarray) -> float:
    ref = ref.reshape((1, 2))

    dst = np.sqrt(np.sum((positions - ref)**2, axis=-1))
    # print(dst.shape)

    influence = np.sum(smoothing_kernel(dst))
    return round(influence * MASS, 6) * SCALING_FACTOR_DENSITY
    
def calculate_pressure_force(particules: list[particule], pos) -> Vector2:
    if not isinstance(pos, Vector2):
        pos = Vector2(pos)

    influence = Vector2(0, 0)
    for p in particules:
        dir = (p.pos - pos)
        dst = dir.magnitude()

        if dst > 0:
            slope = smoothing_kernel_derivative(dst)
            density = p.density
            pressure = density_to_pressure(density)
            dir = dir.elementwise() * pressure * slope * MASS * SCALING_FACTOR_DENSITY / dst * density
            influence += dir
    
    return round(influence, 6)


def density_to_pressure(density: float) -> float:
    return (TARGET_DENSITY - density) * PRESSURE_FACTOR