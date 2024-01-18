from constants import *
import numpy as np
from numba import njit, jit
from pygame import Color, draw

@njit
def tank_collision(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    ref = pos - CENTER_TANK_NUMPY
    cond_x = np.abs(ref[:, 0]) +RADIUS >= TANK[2]/2
    cond_y = np.abs(ref[:, 1]) +RADIUS >= TANK[3]/2

    pos[:, 0] = np.where(cond_x,
        CENTER_TANK_NUMPY[0, 0] +(TANK[2]/2 -RADIUS -0.1) * np.sign(ref[:, 0]),
        pos[:, 0]
    )
    vel[:, 0] = np.where(cond_x, vel[:, 0] * (-0.7), vel[:, 0])

    pos[:, 1] = np.where(cond_y,
        CENTER_TANK_NUMPY[0, 1] +(TANK[3]/2 -RADIUS -0.1) * np.sign(ref[:, 1]),
        pos[:, 1]
    )
    vel[:, 1] = np.where(cond_y, vel[:, 1] * (-0.7), vel[:, 1])

    return pos, vel

@jit
def get_density_color(density: float) -> Color:
    value = density - TARGET_DENSITY
    ref = 0.01

    if abs(value) < ref: # -ref < value < +ref
        col = Color(0, 0, 0)
    
    elif value >= ref: # ref <= value <= MAX_DENSITY
        aux = np.log(value -ref +1)
        aux = aux / np.log(MAX_DENSITY -ref +1)
        col = Color(0, 0, 0).lerp(COLOR_MORE_ATRIB, aux)

    else: # 0 <= density <= TARGET_DENSITY - ref
        aux = np.log(density +1)
        aux = aux / np.log(TARGET_DENSITY -ref +1)
        col = COLOR_LESS_ATRIB.lerp(Color(0, 0, 0), aux)

    return col

@jit
def get_pressure_color(pressure: float) -> Color:
    ref = 0.05

    if abs(pressure) < ref: # -ref < pressure < +ref
        col = Color(250, 250, 250)
    
    elif pressure >= ref: # ref <= pressure <= ???
        aux = np.log(pressure/ref)
        aux = aux / np.log(1/ref)
        col = Color(250, 250, 250).lerp(COLOR_MORE_ATRIB, aux)

    else: # ??? <= pressure <= -ref
        aux = np.log(-pressure/ref)
        aux = aux / np.log(1/ref)
        col = Color(250, 250, 250).lerp(COLOR_LESS_ATRIB, aux)

    return col

@jit
def get_exemple_color(value: float) -> Color:    
    # -1 <= value <= 1
    return COLOR_LESS_ATRIB.lerp(COLOR_MORE_ATRIB, (1+value)/2)

@jit
def draw_smooth_circle(
    surface, color: Color,
    center=(SMOOTHING_RADIUS, SMOOTHING_RADIUS),
    radius:float=SMOOTHING_RADIUS
) -> None:
    r, g, b, _ = color
    a = 0
    for rad in range(radius, 1, -1):
        col = Color(r, g, b, int(a))
        draw.circle(surface, col, center, rad)
        a += (250/radius)
    return