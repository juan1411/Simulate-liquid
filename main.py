"""
Pygame stuffs: create window, handle events and updating frames
"""

import pygame as pg
import numpy as np
from numba import jit, prange

from warnings import filterwarnings
import sys

from constants import *
from utils import *
from liquid import *
from filter import *

filterwarnings("ignore")

class Engine:

    def __init__(self, num_particules: int = NUM_PARTICULES):
        pg.init()
        self.font = pg.font.SysFont(None, 15)
        self.screen = pg.display.set_mode(WIN_RES, flags = pg.DOUBLEBUF)
        pg.display.set_caption("Fluid Simulator")

        self.is_executing = True
        self.is_running = False
        self.show_bg_color = False
        self.show_gradient = False

        self.mouse_value = 0
        self.mouse_radius = 50

        self.clock = pg.time.Clock()
        self.delta_time = 0.1
        self.time = 0

        self.n_parts = num_particules
        self.positions: np.ndarray = None
        self.pred_pos: np.ndarray = None
        self.hashes: np.ndarray = None

        self.velocities: np.ndarray = None
        self.densities: np.ndarray = None
        self.pressures: np.ndarray = None
        self.viscosities: np.ndarray = None
        self.mouse_force: np.ndarray = None
        self.inicial_setup()

    def inicial_setup(self):
        self.positions = create_particules(self.n_parts, "grid")
        self.pred_pos = np.zeros((self.n_parts, 2), dtype=np.float32)
        self.velocities = np.zeros((self.n_parts, 2), dtype=np.float32)
        self.densities = np.zeros((self.n_parts,), dtype=np.float32)
        self.pressures = np.zeros((self.n_parts, 2), dtype=np.float32)
        self.viscosities = np.zeros((self.n_parts, 2), dtype=np.float32)
        self.mouse_force = np.zeros((self.n_parts, 2), dtype=np.float32)

        self.update_predictions()
        self.update_densities()
        self.update_pressures()
        self.update_viscosities()

    # @jit(parallel=True)
    def render(self):
        self.screen.fill(COLOR_BG)
        
        # NOTE: background color
        if self.show_bg_color:
            for y in range(int(TANK[1]), int(TANK[1] +TANK[3]-4), 5):
                for x in prange(int(TANK[0]), int(TANK[0] +TANK[2]-4), 5):
                    pos = np.array((x+3, y+3))
                    d = calculate_density(self.positions, pos)
                    d = (self.densities + d)/2
                    d = d.mean()
                    # exemp = calculate_exemple(self.positions, self.densities, pos)

                    col = get_density_color(d)
                    # col = get_exemple_color(exemp*1.5)
                    pg.draw.rect(self.screen, col, (x, y, 5, 5))

        # NOTE: visualizing gradient direction
        if self.show_gradient:
            inc = PIX_TO_UN//2 +1 -5
            for y in range(int(TANK[1]), int(TANK[1] +TANK[3]), PIX_TO_UN):
                for x in prange(int(TANK[0]), int(TANK[0] +TANK[2]-PIX_TO_UN+1), PIX_TO_UN):
                    pos = np.array((x +inc, y +inc)).reshape((1, 2))

                    d = calculate_density(self.pred_pos, pos)
                    p = calculate_pressure_force(self.pred_pos, self.densities, pos, d)
                    # grad = calculate_exemple_gradient(self.positions, self.densities, pos)

                    end = pos.ravel() + (p/7_000)
                    # end = pos.ravel() + grad
                    pg.draw.circle(self.screen, (210,210,15), (x +inc, y +inc), 4)
                    pg.draw.line(self.screen, (210,210,15), (x +inc, y +inc), end, 3)

        # NOTE: particules
        # tank = pg.Surface((TANK[2], TANK[3])).convert_alpha()
        for i in prange(self.n_parts):
            pos = self.positions[i]

            # alpha_surf = pg.Surface((2*SMOOTHING_RADIUS, 2*SMOOTHING_RADIUS)).convert_alpha()
            # alpha_surf.fill((0, 0, 0, 0))
            # # col = get_exemple_color(exemple_func(pos))
            # draw_smooth_circle(alpha_surf, COLOR_WATER)
            # blit_pos = pos -SMOOTHING_RADIUS -np.array([TANK[0], TANK[1]])
            # tank.blit(alpha_surf, blit_pos)

            pg.draw.circle(self.screen, COLOR_WATER, pos, RADIUS, 2)

        # mouse = np.array(pg.mouse.get_pos()).reshape((1, 2))
        # space = get_lookup_space(positions_to_hash(mouse, 30)[0])
        # for y in range(int(TANK[1]), int(TANK[1] +TANK[3]), 30):
        #     for x in prange(int(TANK[0]), int(TANK[0] +TANK[2]), 30):
        #         aux = np.array([x, y]).reshape(1, 2)
        #         aux = positions_to_hash(aux, 30)
        #         col = "white"
        #         if aux in space: col = "red"
        #         pg.draw.rect(self.screen, col, (x, y, 30, 30), 1)
        
        # self.screen.blit(tank, (TANK[0], TANK[1]))
        pg.draw.circle(self.screen, "red", pg.mouse.get_pos(), self.mouse_radius, 1)
        pg.draw.rect(self.screen, COLOR_TANK, TANK, 1)

        self.draw_text()
        pg.display.flip()

    def draw_text(self):
        # bloco 1
        fps = self.font.render(f"FPS: {self.clock.get_fps():.0f}", True, "white")
        dt = self.font.render(f"Delta time: {self.delta_time:.2f}", True, "white")
        t = self.font.render(f"Passed Time: {self.time:.2f}", True, "white")

        self.screen.blit(fps, (20, 15))
        self.screen.blit(dt, (20, 30))
        self.screen.blit(t, (20, 45))

        # bloco 2
        num = self.font.render(f"N. Particules: {self.n_parts}", True, "white")
        m = self.font.render(f"Mass: {MASS:.1f}", True, "white")
        sr = self.font.render(f"Smooth Radius: {SMOOTHING_RADIUS:.0f}", True, "white")

        self.screen.blit(num, (130, 15))
        self.screen.blit(m, (130, 30))
        self.screen.blit(sr, (130, 45))

        # bloco 3
        g = self.font.render(f"Gravity: {GRAVITY:.0f}", True, "white")
        p = self.font.render(f"Pressure: {FACTOR_PRESSURE:.1f}", True, "white")
        v = self.font.render(f"Viscosity: {FACTOR_VISCOSITY:.1f}", True, "white")
        
        self.screen.blit(g, (240, 15))
        self.screen.blit(p, (240, 30))
        self.screen.blit(v, (240, 45))

        # bloco 4
        d1 = self.font.render(f"Density 1P: {DENSITY_ONE_UNITY:.1f}", True, "white")
        td = self.font.render(f"T. density: {TARGET_DENSITY:.1f}", True, "white")
        md = self.font.render(f"M. density: {self.densities.mean():.1f}", True, "white")
        
        self.screen.blit(d1, (350, 15))
        self.screen.blit(td, (350, 30))
        self.screen.blit(md, (350, 45))

        # bloco 5
        status = self.font.render(f"Stop: {not self.is_running}", True, "white")
        color = self.font.render(f"Bg Color: {self.show_bg_color}", True, "white")
        grad = self.font.render(f"Gradient: {self.show_gradient}", True, "white")

        self.screen.blit(status, (460, 15))
        self.screen.blit(color, (460, 30))
        self.screen.blit(grad, (460, 45))

    @jit
    def update(self):
        self.delta_time = self.clock.tick() * 0.001

        if self.is_running:
            self.time += self.delta_time
            
            # TODO: too slow, find another way?
            self.update_predictions()
            self.update_densities()
            self.update_pressures()
            self.update_viscosities()
            if self.mouse_value != 0: self.update_mouse_force()
            self.update_velocities()

            self.positions += self.velocities * self.delta_time
            self.positions, self.velocities = tank_collision(self.positions, self.velocities)

    @jit(parallel=True)
    def update_densities(self):
        for i in prange(self.n_parts):
            pos = self.pred_pos[i:i+1, :]
            
            hash_pos = positions_to_hash(pos)
            inds = get_indices(hash_pos, self.hashes)

            density = calculate_density(self.pred_pos[inds], pos)
            self.densities[i] = density

    @jit(parallel=True)
    def update_pressures(self):
        for i in prange(self.n_parts):
            pos = self.pred_pos[i:i+1, :]
            dens = self.densities[i]

            hash_pos = positions_to_hash(pos)
            inds = get_indices(hash_pos, self.hashes)

            pressure = calculate_pressure_force(
                self.pred_pos[inds], self.densities[inds],
                pos, dens, FACTOR_PRESSURE
            )
            self.pressures[i] = pressure

    @jit(parallel=True)
    def update_viscosities(self):
        for i in prange(self.n_parts):
            pos = self.pred_pos[i:i+1, :]
            vel = self.velocities[i:i+1, :]

            hash_pos = positions_to_hash(pos)
            inds = get_indices(hash_pos, self.hashes)

            viscosity = calculate_viscosity_force(
                self.pred_pos[inds], self.velocities[inds],
                pos, vel, FACTOR_VISCOSITY
            )
            self.viscosities[i] = viscosity        

    @jit(parallel=True)
    def update_mouse_force(self):
        mouse_pos = np.array(pg.mouse.get_pos()).reshape((1, 2))
    
        self.mouse_force = calculate_mouse_force(
            mouse_pos, self.pred_pos, self.velocities,
            self.mouse_radius, self.mouse_value
        )

    @jit(parallel=True)
    def update_velocities(self):
        density = np.stack((self.densities, self.densities), axis=1)
        self.velocities[:, 1] += GRAVITY * self.delta_time
        self.velocities += self.pressures * self.delta_time / density
        self.velocities += self.viscosities * self.delta_time / density
        self.velocities += self.mouse_force * self.delta_time / density

    @jit(parallel=True)
    def update_predictions(self):
        self.pred_pos = self.positions + (self.velocities * self.delta_time)
        self.hashes = positions_to_hash(self.pred_pos)

    def handle_events(self):
        global GRAVITY, FACTOR_PRESSURE

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_executing = False

            elif event.type == pg.MOUSEBUTTONDOWN:
                pg.mouse.set_visible(False)
                if event.button == 1: self.mouse_value = FACTOR_MOUSE
                elif event.button == 3: self.mouse_value = -FACTOR_MOUSE

            elif event.type == pg.MOUSEBUTTONUP:
                self.mouse_force = np.zeros((self.n_parts, 2), dtype=np.float32)
                pg.mouse.set_visible(True)
                self.mouse_value = 0

            elif event.type == pg.MOUSEWHEEL:
                if event.y > 0: self.mouse_radius += 2
                else:
                    self.mouse_radius -= 2

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_UP: GRAVITY += 1
                elif event.key == pg.K_DOWN: GRAVITY -= 1

                if event.key == pg.K_RIGHT: FACTOR_PRESSURE += 1
                elif event.key == pg.K_LEFT: FACTOR_PRESSURE -= 1

                if event.key == pg.K_SPACE:
                    self.is_running = not self.is_running

                if event.key == pg.K_RETURN:
                    self.time = 0
                    self.inicial_setup()

                if event.key == pg.K_c:
                    self.show_bg_color = not self.show_bg_color

                if event.key == pg.K_BACKSPACE:
                    self.show_gradient = not self.show_gradient

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


if __name__ == "__main__":
    app = Engine()
    app.run()