from pygame import Vector2, Color
from numpy import pi, array

DEBUG = True

# CONSTANTS
WIN_RES = Vector2((1100, 600))
PIX_TO_UN = 50 # NOTE: pixel to "unity"

NUM_PARTICULES = 720 #360
RADIUS = 2
SMOOTHING_RADIUS = 50 * RADIUS

MASS = 1 # pi * (RADIUS ** 2)
VOLUME = pi * (SMOOTHING_RADIUS ** 4) / 6

GRAVITY = 0
FACTOR_DENSITY = 10_000
DENSITY_ONE_UNITY = (SMOOTHING_RADIUS ** 2) * MASS * FACTOR_DENSITY / VOLUME
TARGET_DENSITY = 5 * DENSITY_ONE_UNITY
FACTOR_SLOPE = 1_000
FACTOR_PRESSURE = 1

MAX_DENSITY = 70 * DENSITY_ONE_UNITY

TANK = (20, 100, WIN_RES.x-20-20, WIN_RES.y-100-20)
CENTER_TANK = Vector2(TANK[0] + TANK[2]/2, TANK[1] + TANK[3]/2)
CENTER_TANK_NUMPY = array(CENTER_TANK[:]).reshape((1, 2))

# COLORS
COLOR_BG = (26, 35, 54)
COLOR_WATER = (43, 106, 240)
COLOR_TANK = (250, 250, 250)

COLOR_MORE_ATRIB = Color(2, 7, 240)
COLOR_LESS_ATRIB = Color(250, 0, 96)
COLOR_ARROWS = (43, 240, 43)