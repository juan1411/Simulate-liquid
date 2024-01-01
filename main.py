"""
Pygame stuffs: create window, handle events and updating frames
"""

import pygame as pg
import sys

WIN_RES = (1100, 600)


class Motor:

    def __init__(self):
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

        pg.display.set_mode(WIN_RES, flags = pg.OPENGL | pg.DOUBLEBUF)
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

    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_running = False
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
    app = Motor()
    app.run()