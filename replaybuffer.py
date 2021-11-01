import math
import numpy as np

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
    def __init__(self, capacity, batch_size, alpha):
        assert batch_size < 100

        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.size = 0
        self.next = 0
        self.data = [None] * capacity
        self.heap = [None] * capacity
        self.partitions = [None] * (capacity // 100 + 1)

    def swap(self, a, b):
        self.data[self.heap[a][1]][5] = b
        self.data[self.heap[b][1]][5] = a

        temp = self.heap[a]
        self.heap[a] = self.heap[b]
        self.heap[b] = temp

    def siftup(self, index):
        while index > 0:
            parent = (index - 1) // 2

            if self.heap[parent][0] < self.heap[index][0]:
                self.swap(index, parent)
                index = parent
            else:
                break

    def siftdown(self, index):
        while index < self.size:
            left_child = 2*index + 1
            right_child = 2*index + 2

            if left_child >= self.size:
                break

            if right_child >= self.size or self.heap[left_child][0] > self.heap[right_child][0]:
                max_child = left_child
            else:
                max_child = right_child

            if self.heap[index][0] < self.heap[max_child][0]:
                self.swap(index, max_child)
                index = max_child
            else:
                break

    def add(self, experience):
        priority = 0
        if self.size > 0:
            priority = self.heap[0][0] + 1

        if self.size < self.capacity:
            self.data[self.next] = experience + [self.next]
            self.heap[self.next] = [priority, self.next]
            self.size += 1

            self.siftup(self.next)
        else:
            heap_index = self.data[self.next][5]
            old = self.heap[heap_index][0]
            self.data[self.next] = experience + [heap_index]
            self.heap[heap_index] = [priority, self.next]
            
            if priority > old:
                self.siftup(heap_index)
            elif priority < old:
                self.siftdown(heap_index)

        self.next = (self.next + 1) % self.capacity

        #assert self.is_heap()

    def update(self, indices, priorities):
        for (index, priority) in zip(indices, priorities):
            heap_index = self.data[index][5]
            old = self.heap[heap_index][0]
            self.heap[heap_index][0] = priority

            if priority > old:
                self.siftup(heap_index)
            elif priority < old:
                self.siftdown(heap_index)

        #assert self.is_heap()

    def is_heap(self):
        for i in range(0, self.size):
            left_child = 2*i + 1
            right_child = 2*i + 2

            if left_child < self.size and self.heap[i][0] < self.heap[left_child][0]:
                return False
            
            if right_child < self.size and self.heap[i][0] < self.heap[right_child][0]:
                return False
        
        return True

    def generate_partition(self, size):        
        index = size // 100
        max_size = (index + 1) * 100

        p_dist = np.power(np.arange(1, max_size + 1), -self.alpha)
        p_dist = p_dist / np.sum(p_dist)

        partition = np.zeros(self.batch_size + 1, dtype=np.int32)
        partition[0] = 0
        partition[self.batch_size] = max_size

        cumulative_sum = 0
        index = 0
        for i in range(1, self.batch_size):
            end = i / self.batch_size
            while cumulative_sum < end:
                cumulative_sum += p_dist[index]
                index += 1
            partition[i] = index

        for i in range(1, len(partition)):
            if partition[i] <= partition[i-1]:
                partition[i] = partition[i-1] + 1

        self.partitions[index] = partition

        return partition

    def mini_batch(self):
        partition = self.partitions[self.size // 100]
        if partition == None:
            partition = self.generate_partition(self.size)

        chosen = [None] * self.batch_size
        for i in range(0, self.batch_size):
            start = partition[i]
            end = min(partition[i + 1], self.size)
            chosen[i] = np.random.choice(np.arange(start, end))

        print(chosen)

        exit(0)

        indices = [self.heap[i][1] for i in chosen]
        data = [self.data[i] for i in indices]

        actions = np.asarray([sample[1] for sample in data], dtype=np.int32)
        rewards = np.asarray([sample[2] for sample in data], dtype=np.float32)
        terminals = np.asarray([sample[3] for sample in data], dtype=bool)

        states = [None] * len(data[0][0])
        next_states = [None] * len(data[0][0])
        for i in range(0, len(states)):
            states[i] = np.asarray([sample[0][i] for sample in data], dtype=np.float32)
            next_states[i] = np.asarray([sample[4][i] for sample in data], dtype=np.float32)

        weights = None

        return states, actions, rewards, terminals, next_states, weights, indices