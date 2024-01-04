"""
Pygame stuffs: create window, handle events and updating frames
"""

import pygame as pg
import numpy as np
from numba import jit, prange

from warnings import filterwarnings
import sys

from constants import *
from liquid import *
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
        self.densities: list[float] = []
        self.pressures: list[Vector2] = []
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

        # TODO: too slow, find another way
        self.update_densities()
        self.update_pressures()

        for i in range(len(self.particules)):
            self.particules[i].vel.y += GRAVITY * self.delta_time
            density = self.particules[i].density
            pressure = self.particules[i].pressure
            pressure_vel = -pressure.elementwise() * self.delta_time / density
            self.particules[i].vel += pressure_vel

            self.particules[i].pos += self.particules[i].vel * self.delta_time
            self.particules[i] = tank_collision(self.particules[i])
            self.particules[i].draw(self.screen)


    @jit(parallel=True)
    def update_densities(self):
        positions = np.concatenate([p.get_pos() for p in self.particules], axis=0)

        # TODO: iterate over all particules is slow
        for i in prange(NUM_PARTICULES):
            pos = self.particules[i].get_pos()
            self.particules[i].density = calculate_density(positions, pos)

    # @jit(parallel=True)
    def update_pressures(self):
        positions = np.concatenate([p.get_pos() for p in self.particules], axis=0)
        densities = np.array([p.density for p in self.particules])

        # TODO: iterate over all particules is slow
        for i in prange(NUM_PARTICULES):
            pos = self.particules[i].get_pos()
            aux = calculate_pressure_force(positions, densities, pos)
            self.particules[i].pressure = Vector2(aux[0], aux[1])

    def handle_events(self):
        global GRAVITY

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_running = False

            elif event.type == pg.MOUSEBUTTONDOWN:
                # density = calculate_density(self.particules, event.pos)
                # print(f'{event.pos} - density: {density:.3f}')
                # gradient_dens = calculate_gradient_density(self.particules, event.pos)
                # print(f'{event.pos} - gradient density: ({gradient_dens.x:.3f}, {gradient_dens.y:.3f})')
                print("Mouse button down")

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


def tank_collision(particule: particule) -> particule:

    ref = particule.pos - CENTER_TANK

    if abs(ref.x) + particule.rad >= TANK[2]/2:
        particule.vel.x *= -0.7
        particule.pos.x = CENTER_TANK.x + (TANK[2]/2 - particule.rad - 0.001) * np.sign(ref.x)

    if abs(ref.y) + particule.rad >= TANK[3]/2:
        particule.vel.y *= -0.7
        particule.pos.y = CENTER_TANK.y + (TANK[3]/2 - particule.rad - 0.001) * np.sign(ref.y)

    return particule


if __name__ == "__main__":
    filterwarnings("ignore")

    app = Engine()
    app.run()