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

filterwarnings("ignore")

class Engine:

    def __init__(self, num_particules: int = NUM_PARTICULES):
        pg.init()
        self.font = pg.font.SysFont(None, 20)
        self.screen = pg.display.set_mode(WIN_RES, flags = pg.DOUBLEBUF)
        pg.display.set_caption("Fluid Simulator")

        self.is_executing = True
        self.is_running = False
        self.show_bg_color = False

        self.clock = pg.time.Clock()
        self.delta_time = 0.1
        self.time = 0

        self.n_parts = num_particules
        self.positions: np.ndarray = None
        self.pred_pos: np.ndarray = None
        self.velocities: np.ndarray = None
        self.densities: np.ndarray = None
        self.pressures: np.ndarray = None
        self.inicial_setup()

    def inicial_setup(self):
        self.positions = create_particules(self.n_parts)
        self.pred_pos = np.zeros((self.n_parts, 2), dtype=np.float32)
        self.velocities = np.zeros((self.n_parts, 2), dtype=np.float32)
        self.densities = np.zeros((self.n_parts,), dtype=np.float32)
        self.pressures = np.zeros((self.n_parts, 2), dtype=np.float32)

        self.update_predictions()
        self.update_densities()
        self.update_pressures()

    def render(self):
        self.screen.fill(COLOR_BG)
        
        # NOTE: background color
        if self.show_bg_color:
            for y in range(int(TANK[1]), int(TANK[1] +TANK[3]-4), 5):
                for x in range(int(TANK[0]), int(TANK[0] +TANK[2]-4), 5):
                    pos = np.array((x+3, y+3))
                    d = calculate_density(self.positions, pos)
                    d = (self.densities + d)/2
                    d = d.mean()
                    # print("Value d", d)
                    # exemp = calculate_exemple(self.positions, self.densities, pos)

                    col = get_density_color(d)
                    # col = get_exemple_color(exemp*1.5)
                    pg.draw.rect(self.screen, col, (x, y, 5, 5))

        # # NOTE: visualizing gradient direction
        # inc = PIX_TO_UN//2 +1
        # for y in range(int(TANK[1]), int(TANK[1] +TANK[3]), PIX_TO_UN):
        #     for x in range(int(TANK[0]), int(TANK[0] +TANK[2]-PIX_TO_UN+1), PIX_TO_UN):
        #         pos = np.array((x +inc, y +inc)).reshape((1, 2))

        #         d = calculate_density(self.pred_pos, pos)
        #         p = calculate_pressure_force(self.pred_pos, self.densities, pos, d)
        #         # grad = calculate_exemple_gradient(self.positions, self.densities, pos)

        #         end = pos.ravel() + p
        #         # end = pos.ravel() + grad
        #         pg.draw.circle(self.screen, (210,210,15), (x +inc, y +inc), 4)
        #         pg.draw.line(self.screen, (210,210,15), (x +inc, y +inc), end, 3)

        # # NOTE: tentativa de aproximar o valor da funcao
        # for i in range(self.n_parts):
        #     pos = self.positions[i:i+1, :]

        #     alpha_surf = pg.Surface((2*SMOOTHING_RADIUS, 2*SMOOTHING_RADIUS)).convert_alpha()
        #     alpha_surf.fill((0, 0, 0, 0))
        #     col = get_exemple_color(exemple_func(pos))
        #     draw_smooth_circle(alpha_surf, col)
        #     blit_pos = pos - SMOOTHING_RADIUS
        #     self.screen.blit(alpha_surf, blit_pos.ravel())

        # NOTE: particules
        for i in range(self.n_parts):
            pos = self.positions[i]
            # pre = self.pressures[i] * 0.5
            # pg.draw.line(self.screen, COLOR_ARROWS, pos, pos+pre)
            pg.draw.circle(self.screen, (250,250,250), pos, RADIUS, 2)
        
        pg.draw.circle(self.screen, "green", pg.mouse.get_pos(), SMOOTHING_RADIUS, 1)
        pg.draw.rect(self.screen, COLOR_TANK, TANK, 1)

        self.draw_text()
        pg.display.flip()

    def draw_text(self):
        # bloco 1
        fps = self.font.render(f"FPS: {self.clock.get_fps():.0f}", True, "white")
        dt = self.font.render(f"Delta time: {self.delta_time:.2f}", True, "white")
        t = self.font.render(f"Passed Time: {self.time:.2f}", True, "white")

        self.screen.blit(fps, (20, 20))
        self.screen.blit(dt, (20, 40))
        self.screen.blit(t, (20, 60))

        # bloco 2
        num = self.font.render(f"Num. Particules: {self.n_parts}", True, "white")
        g = self.font.render(f"Gravity: {GRAVITY:.0f}", True, "white")
        p = self.font.render(f"Pressure: {FACTOR_PRESSURE:.1f}", True, "white")

        self.screen.blit(num, (220, 20))
        self.screen.blit(g, (220, 40))
        self.screen.blit(p, (220, 60))

        # bloco 3
        d1 = self.font.render(f"Density 1P: {DENSITY_ONE_UNITY:.1f}", True, "white")
        td = self.font.render(f"T. density: {TARGET_DENSITY:.1f}", True, "white")
        md = self.font.render(f"M. density: {self.densities.mean():.1f}", True, "white")
        
        self.screen.blit(d1, (420, 20))
        self.screen.blit(td, (420, 40))
        self.screen.blit(md, (420, 60))

        # bloco 4
        status = self.font.render(f"Stop: {not self.is_running}", True, "white")
        color = self.font.render(f"Bg Color: {self.show_bg_color}", True, "white")

        self.screen.blit(status, (620, 20))
        self.screen.blit(color, (620, 40))

    # @jit(cache=not DEBUG)
    def update(self):
        self.delta_time = self.clock.tick() * 0.001

        if self.is_running:
            self.time += self.delta_time
            
            # TODO: too slow, find another way?
            self.update_predictions()
            self.update_densities()
            self.update_pressures()
            self.update_velocities()

            self.positions += self.velocities * self.delta_time
            self.positions, self.velocities = tank_collision(self.positions, self.velocities)

    @jit(parallel=True)
    def update_densities(self):
        # TODO: iterate over all particules is slow, filter!
        for i in prange(self.n_parts):
            pos = self.pred_pos[i:i+1, :]
            density = calculate_density(self.pred_pos, pos)
            self.densities[i] = density

    @jit(parallel=True)
    def update_pressures(self):
        # TODO: iterate over all particules is slow, filter!
        for i in prange(self.n_parts):
            pos = self.pred_pos[i:i+1, :]
            dens = self.densities[i]
            pressure = calculate_pressure_force(self.pred_pos, self.densities, pos, dens, FACTOR_PRESSURE)
            self.pressures[i] = pressure

    @jit(parallel=True)
    def update_velocities(self):
        self.velocities[:, 1] += GRAVITY * self.delta_time
        self.velocities[:, 0] += self.pressures[:, 0] * self.delta_time / self.densities
        self.velocities[:, 1] += self.pressures[:, 1] * self.delta_time / self.densities

    @jit(parallel=True)
    def update_predictions(self):
        self.pred_pos = self.positions + (self.velocities * self.delta_time)

    def handle_events(self):
        global GRAVITY, FACTOR_PRESSURE

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_executing = False

            elif event.type == pg.MOUSEBUTTONDOWN:
                print("Mouse button down")

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_UP: GRAVITY += 1
                elif event.key == pg.K_DOWN: GRAVITY -= 1

                if event.key == pg.K_RIGHT: FACTOR_PRESSURE += 0.5
                elif event.key == pg.K_LEFT: FACTOR_PRESSURE -= 0.5

                if event.key == pg.K_SPACE:
                    self.is_running = not self.is_running

                if event.key == pg.K_RETURN:
                    self.time = 0
                    self.inicial_setup()

                if event.key == pg.K_c:
                    self.show_bg_color = not self.show_bg_color

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


# @njit
def tank_collision(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    ref = pos - CENTER_TANK_NUMPY
    cond_x = abs(ref[:, 0]) +RADIUS >= TANK[2]/2
    cond_y = abs(ref[:, 1]) +RADIUS >= TANK[3]/2

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
def get_density_color(density: float) -> pg.Color:
    value = density - TARGET_DENSITY
    ref = 0.01

    if abs(value) < ref: # -ref < value < +ref
        col = pg.Color(0, 0, 0)
    
    elif value >= ref: # ref <= value <= MAX_DENSITY
        aux = np.log(value -ref +1)
        aux = aux / np.log(MAX_DENSITY -ref +1)
        col = pg.Color(0, 0, 0).lerp(COLOR_MORE_ATRIB, aux)

    else: # 0 <= density <= TARGET_DENSITY - ref
        aux = np.log(density +1)
        aux = aux / np.log(TARGET_DENSITY -ref +1)
        col = COLOR_LESS_ATRIB.lerp(pg.Color(0, 0, 0), aux)

    return col

@jit
def get_pressure_color(pressure: float) -> pg.Color:
    ref = 0.05

    if abs(pressure) < ref: # -ref < pressure < +ref
        col = pg.Color(250, 250, 250)
    
    elif pressure >= ref: # ref <= pressure <= ???
        aux = np.log(pressure/ref)
        aux = aux / np.log(1/ref)
        col = pg.Color(250, 250, 250).lerp(COLOR_MORE_ATRIB, aux)

    else: # ??? <= pressure <= -ref
        aux = np.log(-pressure/ref)
        aux = aux / np.log(1/ref)
        col = pg.Color(250, 250, 250).lerp(COLOR_LESS_ATRIB, aux)

    return col

@jit
def get_exemple_color(value: float) -> pg.Color:    
    # -1 <= value <= 1
    return COLOR_LESS_ATRIB.lerp(COLOR_MORE_ATRIB, (1+value)/2)

@njit
def draw_smooth_circle(
    surface, color: pg.Color,
    center=(SMOOTHING_RADIUS, SMOOTHING_RADIUS),
    radius:float=SMOOTHING_RADIUS
) -> None:
    r, g, b, _ = color
    a = 0
    for rad in range(radius, 1, -2):
        col = pg.Color(r, g, b, int(a))
        pg.draw.circle(surface, col, center, rad)
        a += 2*(250/radius)
    return


if __name__ == "__main__":
    app = Engine()
    app.run()