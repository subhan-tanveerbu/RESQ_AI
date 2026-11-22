import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.init()
pygame.font.init()

def render_env(env):
    surface = env.render(cell=42)
    return pygame.image.tostring(surface, "RGB"), surface.get_size()

def render_rgb(env):
    raw, size = render_env(env)
    import numpy as np
    return np.frombuffer(raw, dtype=np.uint8).reshape(size[1], size[0], 3)
