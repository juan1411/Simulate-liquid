from pygame.math import Vector2
from pygame.draw import circle
from numpy import pi
import numpy as np

from constants import *

class particule:

    def __init__(self, pos: tuple | list | Vector2, rad: float = RADIUS):
        self.pos = pos if isinstance(pos, Vector2) else Vector2(pos[0], pos[1])
        self.rad = rad
        self.vel: Vector2 = Vector2(0, 0)
        self.density: float = None
        self.pressure: Vector2 = Vector2(0, 0)
        self.color = COLOR_WATER

    def draw(self, screen):
        circle(screen, self.color, self.pos, self.rad)
    

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
    

def calculate_density(particules: list[particule], pos) -> float:
    if not isinstance(pos, Vector2):
        pos = Vector2(pos)

    influence = 0
    for p in particules:
        dst = (p.pos - pos).magnitude()
        influence += smoothing_kernel(dst)
    
    return round(influence * MASS, 6) * SCALING_FACTOR_DENSITY


def calculate_density_np(particules: list[particule], ref) -> float:
    positions = np.array([ [p.pos.x, p.pos.y] for p in particules])
    ref = np.array(ref).reshape((1, 2))
    # print(positions.shape, pos.shape)

    dst = np.sqrt(np.sum((positions - ref)**2, axis=-1))
    # print(dst.shape)
    return round(np.sum(smoothing_kernel(dst)) * MASS, 6) * SCALING_FACTOR_DENSITY
    
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