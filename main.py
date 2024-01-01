"""
Pygame stuffs: create window, handle events and updating frames
"""

import pygame as pg
import sys

# CONSTANTS
WIN_RES = pg.math.Vector2((1100, 600))
TANK = (5, 5, WIN_RES.x-10, WIN_RES.y-10)
GRAVITY = 0
POSITION = pg.math.Vector2((550, 10))
VELOCITY = 0

# COLORS
COLOR_BG = (26, 35, 54)
COLOR_WATER = (43, 106, 240)
COLOR_TANK = (250, 250, 250)

class Motor:

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

    def render(self):
        pg.display.flip()

    def update(self):
        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks() * 0.001
        pg.display.set_caption(f'FPS: {self.clock.get_fps():.0f} | Time: {self.time:.4f}')

        self.screen.fill(COLOR_BG)

        global POSITION, VELOCITY
        VELOCITY += GRAVITY * self.delta_time
        POSITION.y += VELOCITY * self.delta_time
        POSITION = collision(POSITION)

        pg.draw.circle(self.screen, COLOR_WATER, POSITION, 10)
        pg.draw.rect(self.screen, COLOR_TANK, TANK, 1)

    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_running = False

            elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                global GRAVITY
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


def collision(pos:pg.math.Vector2) -> pg.math.Vector2:
    global VELOCITY

    if abs(pos.x - TANK[2]) > TANK[2]:
        pass

    if abs(pos.y - TANK[3]) > TANK[3]:
        pass

    return pos

if __name__ == "__main__":
    app = Motor()
    app.run()