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

    def smoothing_kernel(self, sample_pos: Vector2) -> float:
        dst = (self.pos - sample_pos).magnitude()
        value = 0

        if dst < SMOOTHING_RADIUS:
           self.color = (240, 106, 43)
           value = SMOOTHING_RADIUS - dst

        return (value ** 3) / VOLUME