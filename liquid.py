from pygame.math import Vector2
from pygame.draw import circle

from constants import *

class particule:

    def __init__(self, pos: tuple | list | Vector2, rad: float = 5):
        self.pos = pos if isinstance(pos, Vector2) else Vector2(pos[0], pos[1])
        self.rad = rad
        self.vel: Vector2 = Vector2(0, 0)

    def draw(self, screen):
        circle(screen, COLOR_WATER, self.pos, self.rad)