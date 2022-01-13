import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk

from .environment import Environment

class ContinuousPathPlanningEnv(Environment):
    """
    Environment for training a path planning agent that can take steps
    of arbitrary length and direction (not locked to grid).
    """

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
        self.path = [np.array([self.start[0], self.start[1]])]
        self.avg_step_len = 0.0

        self.rng = np.random.default_rng()
        self.draw_size = 10
        self.dim = dim
        self.resets = 0

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

    def _is_free(self, pos):
        grid_pos = np.floor(pos).astype(np.int32)

        # Ensure there are two spaces free around pos
        free = True
        for dy in range(-3, 4):
            y = grid_pos[1] + dy
            if y < 0 or y >= self.grid_height:
                free = False
                break

            for dx in range(-3, 4):
                x = grid_pos[0] + dx
                if x < 0 or x >= self.grid_width:
                    free = False
                    break

                if self.grid[y, x] == 1:
                    free = False
                    break

            if not free:
                break

        return free

    def reset(self, start=np.array([0, 0]), goal=np.array([0, 0]), random=True):
        """
        Reset environment to either a random or given state.
        Returns state value for network.
        """

        self.resets += 1
        if random:
            while True:
                self.pos = np.array([self.rng.random() * self.grid_width, self.rng.random() * self.grid_height])
                
                if self._is_free(self.pos):
                    break

            while True:
                self.goal = np.array([self.rng.random() * self.grid_width, self.rng.random() * self.grid_height])
                
                if not np.array_equal(self.pos, self.goal) and self._is_free(self.goal):
                    break
        else:
            self.pos = start
            self.goal = goal

        self.start = self.pos
        self.path = [np.array([self.start[0], self.start[1]])]
        self.avg_step_len = 0.0

        return self.get_state()

    # Source: https://tavianator.com/2011/ray_box.html
    def _intersection(self, pos, dir, dir_norm, dir_inv, grid_pos):
        lx = np.array([grid_pos[0], grid_pos[0] + 1.0])
        ly = np.array([grid_pos[1], grid_pos[1] + 1.0])

        tx = (lx - pos[0]) * dir_inv[0]

        tmin = np.min(tx)
        tmax = np.max(tx)

        ty = (ly - pos[1]) * dir_inv[1]

        tmin = max(tmin, np.min(ty))
        tmax = min(tmax, np.max(ty))
        
        if tmax < tmin or tmin > 1 or tmin < 0:
            return None

        return pos + dir * tmin

    def _raycast(self, pos, dir):
        dir_norm = np.linalg.norm(dir)

        # Just return existing pos if dir's norm is small enough
        if dir_norm < 0.000001:
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

        # Raycasting can miss some edge cases, so let's do a brute force collision test
        dir = new_pos - self.pos
        dir_norm = np.linalg.norm(dir)
        if dir_norm > 0.000001:
            dir_unit = dir / dir_norm
            dist = 0
            new_pos = self.pos
            while dist < dir_norm:
                dist = min(dist + 0.25, dir_norm)
                pos = self.pos + dir_unit * dist
                grid_pos = np.floor(pos).astype(np.int32)

                if (
                    grid_pos[0] < 0 or grid_pos[0] >= self.grid_width or
                    grid_pos[1] < 0 or grid_pos[1] >= self.grid_height or
                    self.grid[grid_pos[1], grid_pos[0]] == 1
                ):
                    hit = True
                    break

                new_pos = pos

        self.pos = new_pos
        self.path.append(new_pos)

        # Keep track of average step length
        dist = np.linalg.norm(self.path[-1] - self.path[-2])
        self.avg_step_len += (dist - self.avg_step_len) / (len(self.path) - 1)

        return not hit

    # Note: this function is only an estimate, not exact
    def _wall_distance(self, pos):
        grid_pos = np.floor(pos).astype(np.int32)

        min_dist = None
        for dy in range(-5, 6):
            y = grid_pos[1] + dy
            for dx in range(-5, 6):
                x = grid_pos[0] + dx

                # If space is a wall
                if (
                    x < 0 or x >= self.grid_width or
                    y < 0 or y >= self.grid_height or
                    self.grid[y, x] == 1
                ):
                    dist = np.linalg.norm((np.array([x, y]) + 0.5) - pos) - 0.5
                    if min_dist is None or dist < min_dist:
                        min_dist = dist

        if not min_dist is None:
            min_dist = max(0.001, min_dist)

        return min_dist

    def step(self, action):
        """
        Take given action and return next state, reward, and whether episode should terminate.
        """

        action = np.squeeze(action)
        result = self._move(action)

        dist = np.linalg.norm(self.goal - self.pos)
        terminal = dist < 0.5

        reward = 0.0

        if terminal:
            reward += 1000

        reward = -dist * 0.05

        # Reward staying away from walls
        wall_dist = self._wall_distance(self.pos)
        if not wall_dist is None:
            reward += 0.5 * max(-0.6 * np.power(wall_dist + 0.5, -1.75), -2.0)

        # Reward component for first order derivative "smoothness".
        # Haven't produced great results yet, needs more work.
        """if len(self.path) >= 3:
            v2 = self.path[-2] - self.path[-3]
            v1 = self.path[-1] - self.path[-2]

            norm = np.linalg.norm(v1)
            if norm > 0.000001:
                v1 /= norm

            norm = np.linalg.norm(v2)
            if norm > 0.000001:
                v2 /= norm

            reward += 0.25 * np.dot(v1, v2)

            # Reward component for second order derivative "smoothness"
            reward += 0.15 * np.linalg.norm(self.path[-3] - 2*self.path[-2] + self.path[-1])

        # Reward component to normalize step length
        dist = np.linalg.norm(self.path[-1] - self.path[-2])
        reward += -0.15 * (self.avg_step_len - dist) ** 2"""

        return self.get_state(), reward, terminal

    def get_state(self):
        """
        Return current state.
        """

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

        # Normalize displacement vector
        dir = self.goal - self.pos
        norm = np.linalg.norm(dir)
        if norm > 0.000001:
            dir = dir / norm

        return [state, dir]

    def draw_img(self, out_file="results/out.png"):
        """
        Draw state of environment as an image.
        """
        
        with Image.new(mode="RGB", size=(self.grid_width*self.draw_size, self.grid_height*self.draw_size)) as img:
            draw = ImageDraw.Draw(img)
            
            for y in range(0, self.grid_height):
                for x in range(0, self.grid_width):
                    color = "#555555" if self.grid[y, x] == 1 else "white"

                    draw.rectangle(
                        xy=[(x*self.draw_size, y*self.draw_size), ((x+1)*self.draw_size, (y+1)*self.draw_size)],
                        #outline=(0, 0, 0),
                        fill=color
                    )

            # Draw start
            ul = (self.start - 0.75) * self.draw_size
            lr = (self.start + 0.75) * self.draw_size
            draw.ellipse([(ul[0], ul[1]), (lr[0], lr[1])], fill="red")

            # Draw goal
            ul = (self.goal - 0.75) * self.draw_size
            lr = (self.goal + 0.75) * self.draw_size
            draw.ellipse([(ul[0], ul[1]), (lr[0], lr[1])], fill="green")
            
            # Draw path taken
            lines = []
            for p in self.path:
                p *= self.draw_size
                lines.append((p[0], p[1]))
            draw.line(lines, width=3, fill="cyan")
            
            img.save(out_file)
    
    def _display_console(self):
        """
        Unimplemented.
        Used to be useful for smaller grids, but lost practicality as grids grew in size.
        """
        return

    def _display_tk(self):
        self.canvas.delete("all")

        # Draw grid spaces
        for y in range(0, self.grid_height):
            for x in range(0, self.grid_width):
                color = "#101010" if self.grid[y, x] == 1 else "white"

                self.canvas.create_rectangle(
                    x*self.draw_size, y*self.draw_size,
                    (x+1)*self.draw_size + 1, (y+1)*self.draw_size + 1,
                    fill=color
                )

        # Draw start
        ul = (self.start - 0.4) * self.draw_size
        lr = (self.start + 0.4) * self.draw_size
        self.canvas.create_oval(ul[0], ul[1], lr[0] + 1, lr[1] + 1, fill="red")

        # Draw goal
        ul = (self.goal - 0.4) * self.draw_size
        lr = (self.goal + 0.4) * self.draw_size
        self.canvas.create_oval(ul[0], ul[1], lr[0] + 1, lr[1] + 1, fill="green")
        
        # Draw path taken (ignore first element, it is the start)
        for p in self.path[1:]:
            ul = (p - 0.25) * self.draw_size
            lr = (p + 0.25) * self.draw_size
            self.canvas.create_oval(ul[0], ul[1], lr[0] + 1, lr[1] + 1, fill="cyan")
        
        # Draw current position
        ul = (self.pos - 0.4) * self.draw_size
        lr = (self.pos + 0.4) * self.draw_size
        self.canvas.create_oval(ul[0], ul[1], lr[0] + 1, lr[1] + 1, fill="orange")

        self.tk_root.update()

    def display(self):
        """
        Display current environment state.
        """

        if self.canvas == None:
            self._display_console()
        else:
            self._display_tk()
        