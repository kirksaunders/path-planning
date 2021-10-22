import numpy as np
from PIL import Image

class PathPlanningEnv:
    def __init__(self, map, dim):
        # Load grid from map
        img = Image.open(map, "r")
        width, height = img.size
        data = img.getdata()
        self.grid_width = width
        self.grid_height = height
        self.grid = np.array(data).reshape((width, height, len(img.getbands())))
        self.grid = self.grid[:, :, 1] # use only first channel
        self.grid = 1 - np.sign(self.grid)

        self.pos = np.array([0, 0])
        self.goal = np.array([0, 0])
        self.path = np.zeros((self.grid_width, self.grid_height), dtype=bool)

        self.num_actions = 8
        self.dim = dim

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

        self.path = np.zeros((self.grid_width, self.grid_height), dtype=bool)

        return self.get_state()

    def move(self, dir):
        new_pos = self.pos + dir

        if (
                new_pos[0] < 0 or new_pos[0] >= self.grid_width or
                new_pos[1] < 0 or new_pos[1] >= self.grid_height or
                self.grid[new_pos[1], new_pos[0]] == 1
           ):
            return False
        else:
            self.path[self.pos[1], self.pos[0]] = True
            self.pos = new_pos
            return True


    def step(self, action):
        old_pos = self.pos

        if action == 0: # up
            result = self.move(np.array([0, -1]))
        elif action == 1: # up right
            result = self.move(np.array([1, -1]))
        elif action == 2: # right
            result = self.move(np.array([1, 0]))
        elif action == 3: # down right
            result = self.move(np.array([1, 1]))
        elif action == 4: # down
            result = self.move(np.array([0, 1]))
        elif action == 5: # down left
            result = self.move(np.array([-1, 1]))
        elif action == 6: # left
            result = self.move(np.array([-1, 0]))
        elif action == 7: # up left
            result = self.move(np.array([-1, -1]))
        else:
            raise ValueError("Invalid action taken: " + str(action))

        reward = -np.linalg.norm(self.goal - self.pos)
        terminal = np.array_equal(self.goal, self.pos)
        if terminal:
            reward += 500
        if result == False:
            reward -= 100

        return self.get_state(), reward, terminal

    def get_state(self):
        state = np.ones((self.dim*2 + 1, self.dim*2 + 1, 1))
        for dy in range(-self.dim, self.dim + 1):
            y = self.pos[1] + dy
            if y >=0 and y < self.grid_height:
                for dx in range(-self.dim, self.dim + 1):
                    x = self.pos[0] + dx
                    if x >= 0 and x < self.grid_width:
                        state[dy+self.dim, dx+self.dim, 0] = self.grid[y, x]

        dir = self.goal - self.pos
        norm = np.linalg.norm(dir)
        if norm == 0:
            norm = 1
        dir = dir / norm

        return [state, dir]
    
    def display(self):
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
