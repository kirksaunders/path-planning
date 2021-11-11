import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk

from env_continuous import *

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
        
        img.save("asd.png")
    env.display()

env = ContinuousPathPlanningEnv("grid2.bmp", DIM, tk_root, on_click_left)
env.reset(random=True)
env.display()

tk_root.mainloop()