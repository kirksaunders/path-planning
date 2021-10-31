from collections import namedtuple
import numpy as np
import random

""" class ReplayBuffer:
    def __init__(self, capacity, state_shapes):
        assert capacity > 0
        self.capacity = capacity
        self.size = 0
        self.insert_index = 0

        assert(type(state_shapes) is list)

        self.state_arrays = [None] * len(state_shapes)
        for i in range(0, len(state_shapes)):
            if type(state_shapes[i]) is tuple:
                shape = (capacity, *(state_shapes[i]))
            else:
                shape = (capacity, state_shapes[i])
            self.state_arrays[i] = np.empty(shape, dtype=np.float32)

        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminal = np.empty(capacity, dtype=bool)
        self.next_state_arrays = copy.deepcopy(self.state_arrays)

    def add(self, state, action, reward, terminal, next_state):
        assert(type(state) is list)
        assert(type(next_state) is list)
        assert(len(state) == len(self.state_arrays))
        assert(len(next_state) == len(self.next_state_arrays))

        for i in range(0, len(state)):
            self.state_arrays[i][self.insert_index, ...] = state[i]
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.terminal[self.insert_index] = terminal
        for i in range(0, len(next_state)):
            self.next_state_arrays[i][self.insert_index, ...] = next_state[i]

        self.insert_index += 1
        if self.insert_index >= self.capacity:
            self.insert_index = 0

        if self.size < self.capacity:
            self.size += 1

    def mini_batch(self, size):
        assert(self.size >= size)

        # Allocate space for return batch
        state_arrays = [None] * len(self.state_arrays)
        for i in range(0, len(self.state_arrays)):
            state_arrays[i] = np.empty((size, *(self.state_arrays[i][0].shape)), dtype=np.float32)
        actions = np.empty(size, dtype=np.int32)
        rewards = np.empty(size, dtype=np.float32)
        terminal = np.empty(size, dtype=bool)
        next_state_arrays = copy.deepcopy(state_arrays)

        # Select samples
        selected = np.random.choice(self.size, size, replace=False)
        for i in range(0, len(selected)):
            for j in range(0, len(state_arrays)):
                state_arrays[j][i] = self.state_arrays[j][selected[i]]
                next_state_arrays[j][i] = self.next_state_arrays[j][selected[i]]
            actions[i] = self.actions[selected[i]]
            rewards[i] = self.rewards[selected[i]]
            terminal[i] = self.terminal[selected[i]]

        return state_arrays, actions, rewards, terminal, next_state_arrays """

class ReplayBuffer:
    def __init__(self, capacity):
        self.data = [None] * capacity
        self.capacity = capacity
        self.size = 0
        self.age = 0
        self.tuple = namedtuple("Experience", ["states", "actions", "rewards", "terminals", "next_states", "priority", "age"])

    def swap(self, a, b, dict_in, dict_out):
        temp = self.data[a]
        self.data[a] = self.data[b]
        self.data[b] = temp

        if dict_in != None and dict_out != None:
            if a in dict_out:
                i = dict_out[a]
                assert dict_in[i] == a
                dict_in[i] = b
                del dict_out[a]
                dict_out[b] = i

            if b in dict_out:
                i = dict_out[b]
                assert dict_in[i] == b
                dict_in[i] = a
                del dict_out[b]
                dict_out[a] = i

    def siftup(self, index, dict_in, dict_out):
        while index > 0:
            parent = (index - 1) // 2

            if self.data[parent][5] < self.data[index][5]:
                self.swap(index, parent, dict_in, dict_out)
                index = parent
            else:
                break

    def siftdown(self, index, dict_in, dict_out):
        while index < self.size:
            left_child = 2*index + 1
            right_child = 2*index + 2

            if self.data[left_child][5] > self.data[right_child][5]:
                max_child = left_child
            else:
                max_child = right_child

            if self.data[index][5] < self.data[max_child][5]:
                self.swap(index, max_child, dict_in, dict_out)
                index = max_child
            else:
                break

    def replace(self, index, value):
        old = self.data[index]
        self.data[index] = value

        if value[5] > old[5]:
            self.siftup(index)
        elif value[5] < old[5]:
            self.siftdown(index)

    def find_old(self):
        old = 0
        index = np.random.choice(2) + 1
        while index < self.size:
            if self.data[index][6] < self.data[old][6]:
                old = index
            index = 2*index + np.random.choice(2) + 1

        return old

    def add(self, experience):
        if self.size < self.capacity:
            self.data[self.size] = experience + (self.data[0][5] + 1, self.age)
            self.size += 1
            self.siftup(self.size - 1)
        else:
            old = self.find_old()
            self.replace(old, experience + (self.data[0][5] + 1, self.age))

        self.age += 1

    def update(self, indices, priorities):
        dict_in = {}
        dict_out = {}
        for i in indices:
            dict_in[i] = i
            dict_out[i] = i

        for i in indices:
            old = self.data[dict_in[i]][5]
            self.data[dict_in[i]][5] = priorities[i]

            if priorities[i] > old:
                self.siftup(i, dict_in, dict_out)
            elif priorities[i] < old:
                self.siftdown(i, dict_in, dict_out)

    def mini_batch(self, size):
        indices = np.random.choice(self.size, size, replace=False)
        data = [self.data[i] for i in indices]
        data = self.tuple(*zip(*data))

        states = [None] * len(data[0][0])
        next_states = [None] * len(data[0][0])
        for i in range(0, len(states)):
            states[i] = np.asarray([x[i] for x in data[0]], dtype=np.float32)
            next_states[i] = np.asarray([x[i] for x in data[4]], dtype=np.float32)

        return states, np.asarray(data[1], dtype=np.int32), np.asarray(data[2], dtype=np.float32), \
               np.asarray(data[3], dtype=bool), next_states, np.asarray(data[5], dtype=np.float32), \
               indices