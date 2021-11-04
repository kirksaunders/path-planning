import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, batch_size, alpha):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.size = 0
        self.tree_size = 0
        self.next = 0
        self.max_priority = 0.1
        self.data = [None] * capacity
        self.tree = [None] * (2*capacity - 1)
        self.rng = np.random.default_rng()

    def _propagate(self, index, delta):
        while index >= 0:
            self.tree[index][0] += delta
            index = (index - 1) // 2

    def _insert_tree(self, value, data_index):
        index = self.tree_size

        if index > 0:
            parent = (index - 1) // 2

            # Move parent to index and put our new value in index+1
            self.tree[index] = [self.tree[parent][0], self.tree[parent][1]]
            self.tree[index + 1] = [value, data_index]

            # Update indices in data table so they point into tree correctly
            self.data[self.tree[index][1]][5] = index
            self.data[data_index][5] = index + 1

            self._propagate(parent, value)
            self.tree_size += 2
        else:
            self.tree[index] = [value, data_index]
            self.tree_size += 1

    def add(self, experience):
        priority = self.max_priority

        if self.size < self.capacity:
            self.data[self.next] = experience + [0]
            self._insert_tree(priority, self.next)
            self.size += 1
        else:
            tree_index = self.data[self.next][5]
            old = self.tree[tree_index][0]
            self.data[self.next] = experience + [tree_index]
            self.tree[tree_index][1] = self.next
            self._propagate(tree_index, priority - old)

        self.next = (self.next + 1) % self.capacity

        #assert self.is_sumtree()

    def update(self, indices, priorities):
        priorities = np.power(priorities + 0.1, self.alpha)
        self.max_priority = max(self.max_priority, np.max(priorities))
        for (index, priority) in zip(indices, priorities):
            tree_index = self.data[index][5]
            old = self.tree[tree_index][0]
            self._propagate(tree_index, priority - old)

        #assert self.is_sumtree()

    def is_sumtree(self):
        for i in range(0, self.tree_size):
            left_child = 2*i + 1
            right_child = left_child + 1

            sum = 0
            leaf = True
            if left_child < self.tree_size:
                leaf = False
                sum += self.tree[left_child][0]

            if right_child < self.tree_size:
                leaf = False
                sum += self.tree[right_child][0]

            if not leaf and abs(self.tree[i][0] - sum) > 0.0001:
                return False
        
        return True

    def sample_tree(self, val):
        assert self.tree_size > 0

        index = 0
        while True:
            left_child = 2*index + 1
            right_child = left_child + 1

            if left_child >= self.tree_size:
                return self.tree[index]
            
            if val <= self.tree[left_child][0] or right_child >= self.tree_size:
                index = left_child
            else:
                index = right_child
                val -= self.tree[left_child][0]

    def mini_batch(self, beta):
        assert self.size >= self.batch_size

        total_sum = self.tree[0][0]

        samples = self.rng.uniform(0.0, total_sum, self.batch_size)
        chosen = [self.sample_tree(i) for i in samples]

        weights = np.asarray([c[0] for c in chosen], dtype=np.float32) / total_sum
        weights = np.power(weights * self.size, -beta)
        weights = weights / np.max(weights)

        indices = [c[1] for c in chosen]
        data = [self.data[i] for i in indices]

        actions = np.asarray([sample[1] for sample in data], dtype=np.int32)
        rewards = np.asarray([sample[2] for sample in data], dtype=np.float32)
        terminals = np.asarray([sample[3] for sample in data], dtype=bool)

        states = [None] * len(data[0][0])
        next_states = [None] * len(data[0][0])
        for i in range(0, len(states)):
            states[i] = np.asarray([sample[0][i] for sample in data], dtype=np.float32)
            next_states[i] = np.asarray([sample[4][i] for sample in data], dtype=np.float32)

        return states, actions, rewards, terminals, next_states, weights, indices