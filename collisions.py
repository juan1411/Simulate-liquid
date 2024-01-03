from liquid import particule
from constants import *

import numpy as np
from pygame.math import Vector2

def tank_collision(particule: particule) -> particule:

    center_tank = Vector2(TANK[0] + TANK[2]/2, TANK[1] + TANK[3]/2)
    ref = particule.pos - center_tank

    if abs(ref.x) + particule.rad >= TANK[2]/2:
        particule.vel.x *= -0.7
        particule.pos.x = center_tank.x + (TANK[2]/2 - particule.rad - 0.001) * np.sign(ref.x)

    if abs(ref.y) + particule.rad >= TANK[3]/2:
        particule.vel.y *= -0.7
        particule.pos.y = center_tank.y + (TANK[3]/2 - particule.rad - 0.001) * np.sign(ref.y)

    return particule