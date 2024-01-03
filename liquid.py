from pygame.math import Vector2
from pygame.draw import circle
from numpy import pi

from constants import *

class particule:

    def __init__(self, pos: tuple | list | Vector2, rad: float = RADIUS):
        self.pos = pos if isinstance(pos, Vector2) else Vector2(pos[0], pos[1])
        self.rad = rad
        self.vel: Vector2 = Vector2(0, 0)
        self.color = COLOR_WATER

    def draw(self, screen):
        circle(screen, self.color, self.pos, self.rad)

    def smoothing_kernel(self, dst: float) -> float:
        value = 0

        if dst < SMOOTHING_RADIUS:
           value = SMOOTHING_RADIUS - dst

        return (value ** 3) / VOLUME
    
    def smoothing_kernel_derivative(self, dst: float) -> float:
        value = 0

        if dst < SMOOTHING_RADIUS:
           value = SMOOTHING_RADIUS - dst

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
        influence += p.smoothing_kernel(dst)
    
    return round(influence * MASS, 6) * SCALING_FACTOR_DENSITY


def calculate_gradient_density(particules: list[particule], pos) -> Vector2:
    if not isinstance(pos, Vector2):
        pos = Vector2(pos)

    influence = Vector2(0, 0)
    for p in particules:
        dir = (p.pos - pos)
        dst = dir.magnitude()

        if dst > 0:
            slope = p.smoothing_kernel_derivative(dst)
            dir = dir.elementwise() * slope * MASS * SCALING_FACTOR_DENSITY / dst
            influence += dir
    
    return round(influence, 6)