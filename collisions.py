from liquid import particule
from constants import *

import numpy as np
from pygame.math import Vector2

def tank_collision(particule: particule) -> particule:

    ref = particule.pos - CENTER_TANK

    if abs(ref.x) + particule.rad >= TANK[2]/2:
        particule.vel.x *= -0.7
        particule.pos.x = CENTER_TANK.x + (TANK[2]/2 - particule.rad - 0.001) * np.sign(ref.x)

    if abs(ref.y) + particule.rad >= TANK[3]/2:
        particule.vel.y *= -0.7
        particule.pos.y = CENTER_TANK.y + (TANK[3]/2 - particule.rad - 0.001) * np.sign(ref.y)

    return particule