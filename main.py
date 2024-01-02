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

    def create_particules(self, num_particules: int = NUM_PARTICULES):
        for _ in range(num_particules):
            pos = np.random.randint(WIN_RES * 0.2, WIN_RES * 0.8, 2)
            self.particules.append(particule(pos))

    def render(self):
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


if __name__ == "__main__":
    app = Engine()
    app.create_particules()
    app.run()