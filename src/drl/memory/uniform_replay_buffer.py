import numpy as np

from .replay_buffer import *

class UniformReplayBuffer(ReplayBuffer):
    """
    Replay buffer that gives all entries equal priority.
    """

    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.size = 0
        self.next = 0
        self.data = [None] * capacity
        self.rng = np.random.default_rng()

    def add(self, experience):
        self.data[self.next] = experience
        
        self.next = (self.next + 1) % self.capacity

        if self.size < self.capacity:
            self.size += 1

    def mini_batch(self):
        indices = self.rng.choice(self.size, self.batch_size, replace=False)
        data = [self.data[i] for i in indices]

        actions = np.asarray([sample[1] for sample in data], dtype=np.float32)
        rewards = np.asarray([sample[2] for sample in data], dtype=np.float32).reshape((self.batch_size, 1))
        terminals = np.asarray([sample[3] for sample in data], dtype=bool).reshape((self.batch_size, 1))

        states = [None] * len(data[0][0])
        next_states = [None] * len(data[0][0])
        for i in range(0, len(states)):
            states[i] = np.asarray([sample[0][i] for sample in data], dtype=np.float32)
            next_states[i] = np.asarray([sample[4][i] for sample in data], dtype=np.float32)

        return states, actions, rewards, terminals, next_states