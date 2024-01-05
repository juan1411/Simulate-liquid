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


class Engine:

    def __init__(self):
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

        self.screen = pg.display.set_mode(WIN_RES, flags = pg.DOUBLEBUF)
        self.is_executing = True
        self.is_running = False

        self.clock = pg.time.Clock()
        self.delta_time = 1
        self.time = 0

        self.positions: list[np.ndarray] = []
        self.velocities: list[Vector2] = []
        self.densities: list[float] = []
        self.pressures: list[Vector2] = []
        self.inicial_setup()

    def inicial_setup(self):
        self.positions = create_particules(NUM_PARTICULES, "grid")
        for _ in range(NUM_PARTICULES):
            self.velocities.append(Vector2(0, 0))
            self.densities.append(0)
            self.pressures.append(Vector2(0, 0))

        self.update_densities()
        self.update_pressures()
        for i in range(NUM_PARTICULES):
            self.update_velocities(i)

    def render(self):
        pg.display.set_caption(f'FPS: {self.clock.get_fps():.0f} | Time: {self.time:.4f}')
        self.screen.fill(COLOR_BG)
        alpha_surf = pg.Surface(WIN_RES, pg.SRCALPHA)

        for i in range(NUM_PARTICULES):
            a = self.positions[i]
            p = self.pressures[i].elementwise() * 5
            pg.draw.line(self.screen, COLOR_ARROWS, (a[0], a[1]), (a[0] + p.x, a[1] + p.y))
            pg.draw.circle(alpha_surf, COLOR_PRES, (a[0], a[1]), SMOOTHING_RADIUS)
            pg.draw.circle(self.screen, COLOR_WATER, (a[0], a[1]), RADIUS)

        self.screen.blit(alpha_surf, (0, 0))
        pg.draw.circle(self.screen, "green", pg.mouse.get_pos(), SMOOTHING_RADIUS, 1)
        pg.draw.rect(self.screen, COLOR_TANK, TANK, 1)
        pg.display.flip()

    def update(self):
        self.delta_time = self.clock.tick() * 0.001

        if self.is_running:
            self.time += self.delta_time

            # TODO: too slow, find another way?
            self.update_densities()
            self.update_pressures()

            for i in prange(NUM_PARTICULES):
                self.update_velocities(i)
                dir = self.velocities[i] * self.delta_time
                self.positions[i] = np.add(self.positions[i], np.array(dir[:]), dtype=np.float32)
                self.positions[i], self.velocities[i] = tank_collision(self.positions[i], self.velocities[i])


    def update_velocities(self, index: int):
        self.velocities[index].y += GRAVITY * self.delta_time
        pressure_vel = self.pressures[index].elementwise() * self.delta_time / self.densities[index]
        self.velocities[index] += pressure_vel

    @jit(parallel=True, cache=True)
    def update_densities(self):
        aux = np.array(self.positions, dtype=np.float32).reshape((NUM_PARTICULES, 2))

        # TODO: iterate over all particules is slow, filter!
        for i in prange(NUM_PARTICULES):
            pos = self.positions[i]
            self.densities[i] = calculate_density(aux, pos)

    @jit(parallel=True, cache=True)
    def update_pressures(self):
        aux_pos = np.array(self.positions, dtype=np.float32).reshape((NUM_PARTICULES, 2))
        aux_den = np.array(self.densities, dtype=np.float32)

        # TODO: iterate over all particules is slow, filter!
        for i in prange(NUM_PARTICULES):
            pos = self.positions[i]
            pressure = calculate_pressure_force(aux_pos, aux_den, pos)
            self.pressures[i] = Vector2(pressure[0], pressure[1])

    def handle_events(self):
        global GRAVITY

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_executing = False

            elif event.type == pg.MOUSEBUTTONDOWN:
                # density = calculate_density(self.particules, event.pos)
                # print(f'{event.pos} - density: {density:.3f}')
                # gradient_dens = calculate_gradient_density(self.particules, event.pos)
                # print(f'{event.pos} - gradient density: ({gradient_dens.x:.3f}, {gradient_dens.y:.3f})')
                print("Mouse button down")

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_UP: GRAVITY += 1
                elif event.key == pg.K_DOWN: GRAVITY -= 1

                if event.key in (pg.K_SPACE, pg.K_KP_ENTER):
                    self.is_running = not self.is_running

            else:
                if hasattr(event, "key"):
                    print(f'Type: {pg.event.event_name(event.type)} - key: {pg.key.name(event.key)}')

    def run(self):
        while self.is_executing:
            self.handle_events()
            self.update()
            self.render()

        pg.quit()
        sys.exit()


def tank_collision(pos: np.ndarray, vel: Vector2) -> tuple[np.ndarray, Vector2]:

    new_pos = Vector2(pos[0], pos[1])
    ref = new_pos - CENTER_TANK

    if abs(ref.x) + RADIUS >= TANK[2]/2:
        vel.x *= -0.7
        new_pos.x = CENTER_TANK.x + (TANK[2]/2 - RADIUS - 0.001) * np.sign(ref.x)

    if abs(ref.y) + RADIUS >= TANK[3]/2:
        vel.y *= -0.7
        new_pos.y = CENTER_TANK.y + (TANK[3]/2 - RADIUS - 0.001) * np.sign(ref.y)

    return np.array(new_pos[:]), vel


if __name__ == "__main__":
    filterwarnings("ignore")

    app = Engine()
    app.run()