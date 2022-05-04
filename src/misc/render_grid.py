# Renders grids as bigger PNG images.
# Run from project root with command: python -m src.misc.render_grid

import numpy as np
from PIL import Image, ImageDraw

maps = ["grid_general2", "west_green_small"]
draw_size = 10

for map in maps:
    # Load grid from map
    img = Image.open("grids/" + map + ".bmp", "r")
    width, height = img.size
    data = img.getdata()
    grid_width = width
    grid_height = height
    grid = np.array(data).reshape((height, width, len(img.getbands())))
    grid = grid[:, :, 1] # use only first channel
    grid = 1 - np.sign(grid)

    with Image.new(mode="RGB", size=(grid_width*draw_size, grid_height*draw_size)) as img:
        draw = ImageDraw.Draw(img)
        
        for y in range(0, grid_height):
            for x in range(0, grid_width):
                color = "#000000" if grid[y, x] == 1 else "white"

                draw.rectangle(
                    xy=[(x*draw_size, y*draw_size), ((x+1)*draw_size, (y+1)*draw_size)],
                    #outline=(0, 0, 0),
                    fill=color
                )

        img.save("docs-new/graphics/" + map + ".png")