"""
Pygame stuffs: create window, handle events and updating frames
"""

import pygame as pg
import numpy as np
import sys

from constants import *
from liquid import particule
from collisions import *


class Engine:

    def __init__(self):
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

        self.screen = pg.display.set_mode(WIN_RES, flags = pg.DOUBLEBUF)
        self.is_running = True

        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.time = 0

        self.particules:list[particule] = []
        self.inicial_setup()

    def inicial_setup(self):
        self.particules = create_particules()

    def render(self):
        pg.draw.circle(self.screen, "green", pg.mouse.get_pos(), SMOOTHING_RADIUS, 1)
        pg.draw.rect(self.screen, COLOR_TANK, TANK, 1)
        pg.display.flip()

    def update(self):
        self.delta_time = self.clock.tick() * 0.001
        self.time = pg.time.get_ticks() * 0.001
        pg.display.set_caption(f'FPS: {self.clock.get_fps():.0f} | Time: {self.time:.4f}')

        self.screen.fill(COLOR_BG)

        for i in range(len(self.particules)):
            self.particules[i].vel.y += GRAVITY * self.delta_time
            self.particules[i].pos += self.particules[i].vel * self.delta_time

            self.particules[i] = tank_collision(self.particules[i])
            self.particules[i].draw(self.screen)
        

    def handle_events(self):
        global GRAVITY

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_running = False

            elif event.type == pg.MOUSEBUTTONDOWN:
                # density = calculate_density(self.particules, event.pos)
                # print(f'{event.pos} - density: {density:.3f}')
                gradient_dens = calculate_gradient_density(self.particules, event.pos)
                print(f'{event.pos} - gradient density: ({gradient_dens.x:.3f}, {gradient_dens.y:.3f})')

            elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                GRAVITY += 1

            else:
                if hasattr(event, "key"):
                    print(f'Type: {pg.event.event_name(event.type)} - key: {pg.key.name(event.key)}')

    def run(self):
        while self.is_running:
            self.handle_events()
            self.update()
            self.render()

        pg.quit()
        sys.exit()


def create_particules(num_particules: int = NUM_PARTICULES) -> list[particule]:
    spacing = 7
    per_row = 130
    particules = []
    for i in range(num_particules):
        # pos = np.random.randint(WIN_RES * 0.1, WIN_RES * 0.9, 2)
        pos = (100 + (i%per_row - 1) * spacing, 75 + (i//per_row + 1) * spacing)
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


if __name__ == "__main__":
    app = Engine()
    app.run()