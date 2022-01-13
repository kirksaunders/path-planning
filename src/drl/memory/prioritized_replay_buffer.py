from abc import abstractmethod
import numpy as np

from .replay_buffer import *

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Abstract replay buffer that prioritizes certain entries over others.
    See paper (https://arxiv.org/abs/1511.05952).
    """

    @abstractmethod
    def update(self, indices, priorities):
        pass