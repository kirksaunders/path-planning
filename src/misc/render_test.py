# Allows the user to act as an agent by clicking and renders useful
# graphics to the current working directory as PNGs.
# Run from project root with command: python -m src.misc.render_test

import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk

from ..drl.environments.continuous_path_planning import *

DIM = 5
NUM_FRAMES = 1
RENDER_SIZE = 25
EXTRA = 5

tk_root = tk.Tk()

input = 0

def on_click_left(event):
    x = event.x / env.draw_size
    y = event.y / env.draw_size

    # Get dir in local space
    dir = np.array([x, y]) - env.pos
    dir = np.matmul(dir, env.axis_matrix_inv)
    
    frames, reward, terminal = env.step(dir)
    state = [x[-1, ...] for x in frames]
    print(reward, terminal)
    print(state[1])
    with Image.new(mode="RGB", size=((2*DIM+1)*RENDER_SIZE, (2*DIM+1)*RENDER_SIZE)) as img:
        draw = ImageDraw.Draw(img)
        
        for y in range(0, 2*DIM+1):
            for x in range(0, 2*DIM+1):
                color = int((1.0 - state[0][y, x]) * 255)
                draw.rectangle(
                    xy=[(x*RENDER_SIZE, y*RENDER_SIZE), ((x+1)*RENDER_SIZE, (y+1)*RENDER_SIZE)],
                    outline=(0, 0, 0),
                    fill=(color, color, color)
                )
        
        img.save("input.png")

    grid_pos = np.floor(env.pos).astype(np.int32)

    with Image.new(mode="RGB", size=((2*(DIM+EXTRA)+1)*RENDER_SIZE, (2*(DIM+EXTRA)+1)*RENDER_SIZE)) as img:
        draw = ImageDraw.Draw(img)

        for dy in range(-DIM - EXTRA, DIM + EXTRA + 1):
            y = grid_pos[1] + dy
            for dx in range(-DIM - EXTRA, DIM + EXTRA + 1):
                x = grid_pos[0] + dx
                if x >= 0 and x < env.grid_width and y >= 0 and y < env.grid_height:
                    color = int((1.0 - env.grid[y, x]) * 255)
                else:
                    color = 0
                x2 = dx+EXTRA+DIM
                y2 = dy+EXTRA+DIM
                draw.rectangle(
                    xy=[(x2*RENDER_SIZE, y2*RENDER_SIZE), ((x2+1)*RENDER_SIZE, (y2+1)*RENDER_SIZE)],
                    outline=(0, 0, 0),
                    fill=(color, color, color)
                )

        vert_axis = -env.axis
        horiz_axis = np.array([vert_axis[1], -vert_axis[0]])

        center = np.array([(2*(DIM+EXTRA)+1)/2, (2*(DIM+EXTRA)+1)/2])

        dif = env.pos - (grid_pos + 0.5)
        ul = (center + dif + (DIM + 0.5) * (-vert_axis - horiz_axis)) * RENDER_SIZE
        ur = (center + dif + (DIM + 0.5) * (-vert_axis + horiz_axis)) * RENDER_SIZE
        lr = (center + dif + (DIM + 0.5) * (vert_axis + horiz_axis)) * RENDER_SIZE
        ll = (center + dif + (DIM + 0.5) * (vert_axis - horiz_axis)) * RENDER_SIZE

        draw.polygon(
            [(ul[0], ul[1]), (ur[0], ur[1]), (lr[0], lr[1]), (ll[0], ll[1])],
            outline=(175, 100, 50),
            width=5,
            fill=None
        )

        ul = (center + dif - 0.25) * RENDER_SIZE
        lr = (center + dif + 0.25) * RENDER_SIZE
        draw.ellipse([(ul[0], ul[1]), (lr[0], lr[1])], fill="cyan")
        
        img.save("overview.png")
    env.display()

env = ContinuousPathPlanningEnv("grids/grid2.bmp", DIM, NUM_FRAMES, tk_root, on_click_left)
env.reset(random=True)
env.display()

tk_root.mainloop()