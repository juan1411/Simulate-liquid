from liquid import particule
from constants import *

import numpy as np
from pygame.math import Vector2

def tank_collision(particule: particule) -> particule:

    ref = particule.pos - WIN_RES/2 + Vector2(particule.rad, particule.rad)

    if abs(ref.x) >= TANK[2]/2:
        particule.vel.x *= -0.7
        particule.pos.x = WIN_RES.x/2 + (TANK[2]/2 - particule.rad - 0.001) * np.sign(ref.x)

    if abs(ref.y) >= TANK[3]/2:
        particule.vel.y *= -0.7
        particule.pos.y = WIN_RES.y/2 + (TANK[3]/2 - particule.rad - 0.001) * np.sign(ref.y)

    return particule