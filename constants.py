from pygame import Vector2
from numpy import pi

# CONSTANTS
WIN_RES = Vector2((1100, 600))
TANK = (20, 100, WIN_RES.x-20-20, WIN_RES.y-100-20)
CENTER_TANK = Vector2(TANK[0] + TANK[2]/2, TANK[1] + TANK[3]/2)

GRAVITY = 0
TARGET_DENSITY = 0.5
PRESSURE_FACTOR = 50

NUM_PARTICULES = 360
RADIUS = 10
SMOOTHING_RADIUS = 50

MASS = pi * (RADIUS ** 2)
VOLUME = pi * (SMOOTHING_RADIUS ** 5) / 10
SCALING_FACTOR_DENSITY = 100

# COLORS
COLOR_BG = (26, 35, 54)
COLOR_WATER = (43, 106, 240)
COLOR_TANK = (250, 250, 250)