import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk

class ContinuousPathPlanningEnv:
    def __init__(self, map, dim, tkinter_root=None, on_click_left=None, on_click_right=None):
        # Load grid from map
        img = Image.open(map, "r")
        width, height = img.size
        data = img.getdata()
        self.grid_width = width
        self.grid_height = height
        self.grid = np.array(data).reshape((height, width, len(img.getbands())))
        self.grid = self.grid[:, :, 1] # use only first channel
        self.grid = 1 - np.sign(self.grid)

        self.pos = np.array([0, 0])
        self.start = np.array([0, 0])
        self.goal = np.array([0, 0])
        self.path = []

        self.draw_size = 15
        self.dim = dim

        if tkinter_root != None:
            self.tk_root = tkinter_root
            self.canvas = tk.Canvas(tkinter_root, width=self.grid_width*self.draw_size, height=self.grid_height*self.draw_size)
            if on_click_left != None:
                self.canvas.bind("<Button-1>", on_click_left)
            if on_click_right != None:
                self.canvas.bind("<Button-3>", on_click_right)
            self.canvas.pack()
        else:
            self.canvas = None

    def reset(self, start=np.array([0, 0]), goal=np.array([0, 0]), random=False):
        if random:
            while True:
                self.pos = np.array([np.random.choice(self.grid_width), np.random.choice(self.grid_height)])
                if self.grid[self.pos[1], self.pos[0]] == 0:
                    break

            while True:
                self.goal = np.array([np.random.choice(self.grid_width), np.random.choice(self.grid_height)])
                if self.grid[self.goal[1], self.goal[0]] == 0 and not np.array_equal(self.goal, self.pos):
                    break
        else:
            self.pos = start
            self.goal = goal

        self.start = self.pos
        self.pos = self.pos + 0.5
        self.path = []

        return self.get_state()

    # Source: https://tavianator.com/2011/ray_box.html
    def _intersection(self, pos, dir, dir_norm, dir_inv, grid_pos):
        lx = np.array([grid_pos[0], grid_pos[0] + 1])
        ly = np.array([grid_pos[1], grid_pos[1] + 1])

        tx = (lx - pos[0]) * dir_inv[0]

        tmin = np.min(tx)
        tmax = np.max(tx)

        ty = (ly - pos[1]) * dir_inv[1]

        tmin = max(tmin, np.min(ty))
        tmax = min(tmax, np.max(ty))
        
        if tmax < tmin or tmin > 1 or tmin < 0:
            return None

        # Ensure we stay at least 0.1 distance away from wall
        dist = tmin - 0.1 / dir_norm
        return pos + dir * dist

    def _raycast(self, pos, dir):
        dir_norm = np.linalg.norm(dir)

        # Just return existing pos if dir's norm is small enough
        if abs(dir_norm) < 0.000001:
            return pos, False

        # Pre-calculate dir_inv to save some time
        dir_inv = 1.0 / dir

        # Search outward from grid squares for intersection
        min_intersection = None
        grid_pos = np.floor(pos).astype(np.int32)
        spiral_dir = np.array([1, 0], dtype=np.int32)
        spiral_length = 1
        spiral_steps = 0
        while True:
            # Spiral outward from original start grid pos
            grid_pos += spiral_dir
            spiral_steps += 1
            if spiral_steps == spiral_length:
                spiral_steps = 0

                # Rotate direction counter-clockwise
                tmp = spiral_dir[1]
                spiral_dir[1] = -spiral_dir[0]
                spiral_dir[0] = tmp

                # If we made a turn from vertical to horizontal, length increases
                if spiral_dir[1] == 0:
                    spiral_length += 1

                # If dir is [0, -1], we made a full loop and the radius has increased.
                # Since radius increased, we don't need to keep searching if we already
                # found an intersection, it should be the closest one
                if spiral_dir[0] == 0 and spiral_dir[1] == -1 and not min_intersection is None:
                    return min_intersection, True

            grid_center = grid_pos + 0.5

            # Quit if we can't possibly reach any more grid spaces
            if np.linalg.norm(grid_center - pos) - 1 > dir_norm:
                break

            # Check for intersection and return if found
            if (
                grid_pos[0] < 0 or grid_pos[0] >= self.grid_width or
                grid_pos[1] < 0 or grid_pos[1] >= self.grid_height or
                self.grid[grid_pos[1], grid_pos[0]] == 1
            ):
                intersection = self._intersection(pos, dir, dir_norm, dir_inv, grid_pos)
                if not intersection is None:
                    if not min_intersection is None:
                        dpos_new = intersection - pos
                        dpos_old = min_intersection - pos
                        if np.dot(dpos_new, dpos_new) < np.dot(dpos_old, dpos_old):
                            min_intersection = intersection
                    else:
                        min_intersection = intersection

        if min_intersection is None:
            # No intersection found, just advance position
            return pos + dir, False
        else:
            return min_intersection, True

    def _move(self, dir):
        new_pos, hit = self._raycast(self.pos, dir)
        self.pos = new_pos
        self.path.append(new_pos)

        return not hit

    def step(self, action):
        result = self._move(action)

        terminal = np.linalg.norm((self.goal + 0.5) - self.pos) < 0.5
        reward = 0
        if terminal:
            reward += 1
        else:
            reward -= 1
        if result == False:
            reward -= 1

        #dist = np.linalg.norm(self.goal - self.pos)
        #reward = -0.01 * dist * dist - 5
        #terminal = np.array_equal(self.goal, self.pos)
        #if terminal:
        #    reward += 500
        #if result == False:
        #    reward -= 50

        # "Normalize" the reward so it's closer to the range [-1, 1]
        #reward = reward / 1000.0
        # reward clipping?
        #reward = np.sign(reward)

        return self.get_state(), reward, terminal

    def get_state(self):
        grid_pos = np.floor(self.pos).astype(np.int32)

        # Build coeff matrix for performing bilinear interpolation (very similar to a convolution)
        dpos = self.pos - (grid_pos + 0.5)
        dx = dpos[0]
        dy = dpos[1]
        interp = np.zeros((3, 3))
        interp[0, 0] = (-min(dx, 0.0)) * (-min(dy, 0.0))
        interp[0, 1] = (1 - abs(dx)) * (-min(dy, 0.0))
        interp[0, 2] = max(dx, 0.0) * (-min(dy, 0.0))
        interp[1, 0] = (-min(dx, 0.0)) * (1 - abs(dy))
        interp[1, 1] = (1 - abs(dx)) * (1 - abs(dy))
        interp[1, 2] = max(dx, 0.0) * (1 - abs(dy))
        interp[2, 0] = (-min(dx, 0.0)) * max(dy, 0.0)
        interp[2, 1] = (1 - abs(dx)) * max(dy, 0.0)
        interp[2, 2] = max(dx, 0.0) * max(dy, 0.0)

        state = np.ones((self.dim*2 + 1, self.dim*2 + 1, 1))
        for dy in range(-self.dim, self.dim + 1):
            y = grid_pos[1] + dy
            for dx in range(-self.dim, self.dim + 1):
                x = grid_pos[0] + dx

                # Get surrounding grid spaces
                spaces = np.ones((3, 3))
                for dy2 in range(-1, 2):
                    y2 = y + dy2
                    if y2 >= 0 and y2 < self.grid_height:
                        for dx2 in range(-1, 2):
                            x2 = x + dx2
                            if x2 >= 0 and x2 < self.grid_width:
                                spaces[dy2 + 1, dx2 + 1] = self.grid[y2, x2]

                # Apply interpolation coefficients on these spaces
                state[dy+self.dim, dx+self.dim, 0] = np.sum(np.multiply(spaces, interp))

        dir = self.goal - self.pos
        #norm = np.linalg.norm(dir)
        #if norm == 0:
        #    norm = 1
        #dir = dir / norm

        return [state, dir]

    def draw_img(self, out_file="results/out.png"):
        with Image.new(mode="RGB", size=(self.grid_width*self.draw_size, self.grid_height*self.draw_size)) as img:
            draw = ImageDraw.Draw(img)
            
            for y in range(0, self.grid_height):
                for x in range(0, self.grid_width):
                    if x == self.goal[0] and y == self.goal[1]:
                        color = (0, 255, 0)
                    elif x == self.start[0] and y == self.start[1]:
                        color = (255, 0, 0)
                    elif x == self.pos[0] and y == self.pos[1]:
                        color = (225, 100, 25)
                    elif self.grid[y, x] == 1:
                        color = (50, 50, 50)
                    elif self.path[y, x] == 1:
                        color = (150, 150, 255)
                    else:
                        color = (225, 225, 225)

                    draw.rectangle(
                        xy=[(x*self.draw_size, y*self.draw_size), ((x+1)*self.draw_size, (y+1)*self.draw_size)],
                        outline=(0, 0, 0),
                        fill=color
                    )
            
            img.save(out_file)
    
    def _display_console(self):
        for y in range(0, self.grid_height):
            for x in range(0, self.grid_width):
                if x == self.pos[0] and y == self.pos[1]:
                    print("P", end="")
                elif x == self.goal[0] and y == self.goal[1]:
                    print("G", end="")
                elif self.grid[y, x] == 1:
                    print("W", end="")
                elif self.path[y, x] == 1:
                    print("X", end="")
                else:
                    print("+", end="")
            print("")
        print("")
        print("")

    def _display_tk(self):
        self.canvas.delete("all")
        for y in range(0, self.grid_height):
            for x in range(0, self.grid_width):
                if x == self.goal[0] and y == self.goal[1]:
                    color = "green"
                elif x == self.start[0] and y == self.start[1]:
                    color = "red"
                elif self.grid[y, x] == 1:
                    color = "#101010"
                else:
                    color = "white"

                self.canvas.create_rectangle(
                    x*self.draw_size, y*self.draw_size,
                    (x+1)*self.draw_size + 1, (y+1)*self.draw_size + 1,
                    fill=color
                )
        for p in self.path:
            ul = (p - 0.25) * self.draw_size
            lr = (p + 0.25) * self.draw_size
            self.canvas.create_oval(ul[0], ul[1], lr[0] + 1, lr[1] + 1, fill="cyan")
        
        ul = (self.pos - 0.4) * self.draw_size
        lr = (self.pos + 0.4) * self.draw_size
        self.canvas.create_oval(ul[0], ul[1], lr[0] + 1, lr[1] + 1, fill="orange")
        self.tk_root.update()

    def display(self):
        if self.canvas == None:
            self._display_console()
        else:
            self._display_tk()
        
