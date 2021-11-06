from abc import abstractmethod
import numpy as np

from replay_buffer import *

class PrioritizedReplayBuffer(ReplayBuffer):
    @abstractmethod
    def update(self, indices, priorities):
        pass