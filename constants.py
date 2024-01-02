from pygame.math import Vector2
from numpy import pi

# CONSTANTS
WIN_RES = Vector2((1100, 600))
TANK = (20, 20, WIN_RES.x-40, WIN_RES.y-40)
GRAVITY = 00
NUM_PARTICULES = 999
RADIUS = 4
SMOOTHING_RADIUS = 50

MASS = 1
VOLUME = pi * 0.1 * (SMOOTHING_RADIUS ** 5)

# COLORS
COLOR_BG = (26, 35, 54)
COLOR_WATER = (43, 106, 240)
COLOR_TANK = (250, 250, 250)