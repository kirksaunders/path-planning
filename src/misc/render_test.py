# Allows the user to act as an agent by clicking and renders useful
# graphics to the current working directory as PNGs.

import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk

from drl.environments.continuous_path_planning import *

DIM = 5

tk_root = tk.Tk()

input = 0

def on_click_left(event):
    x = event.x / env.draw_size
    y = event.y / env.draw_size
    
    state, reward, terminal = env.step(np.array([x, y]) - env.pos)
    print(reward, terminal)
    with Image.new(mode="RGB", size=((2*DIM+1)*25, (2*DIM+1)*25)) as img:
        draw = ImageDraw.Draw(img)
        
        for y in range(0, 2*DIM+1):
            for x in range(0, 2*DIM+1):
                color = int((1.0 - state[0][y, x, 0]) * 255)
                draw.rectangle(
                    xy=[(x*25, y*25), ((x+1)*25, (y+1)*25)],
                    outline=(0, 0, 0),
                    fill=(color, color, color)
                )
        
        img.save("input.png")

    grid_pos = np.floor(env.pos).astype(np.int32)

    extra = 5

    with Image.new(mode="RGB", size=((2*(DIM+extra)+1)*25, (2*(DIM+extra)+1)*25)) as img:
        draw = ImageDraw.Draw(img)

        for dy in range(-DIM - extra, DIM + extra + 1):
            y = grid_pos[1] + dy
            for dx in range(-DIM - extra, DIM + extra + 1):
                x = grid_pos[0] + dx
                color = int((1.0 - env.grid[y, x]) * 255)
                x2 = dx+extra+DIM
                y2 = dy+extra+DIM
                draw.rectangle(
                    xy=[(x2*25, y2*25), ((x2+1)*25, (y2+1)*25)],
                    outline=(0, 0, 0),
                    fill=(color, color, color)
                )

        center = np.array([(2*(DIM+extra)+1)/2, (2*(DIM+extra)+1)/2])

        dif = env.pos - (grid_pos + 0.5)
        ul = (center + dif - DIM - 0.5) * 25
        lr = (center + dif + DIM + 0.5) * 25

        draw.rectangle(
            xy=[(ul[0], ul[1]), (lr[0], lr[1])],
            outline=(175, 100, 50),
            width=5,
            fill=None
        )

        ul = (center + dif - 0.25) * 25
        lr = (center + dif + 0.25) * 25
        draw.ellipse([(ul[0], ul[1]), (lr[0], lr[1])], fill="cyan")
        
        img.save("overview.png")
    env.display()

env = ContinuousPathPlanningEnv("grid2.bmp", DIM, tk_root, on_click_left)
env.reset(random=True)
env.display()

tk_root.mainloop()