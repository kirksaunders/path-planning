import copy
from env import PathPlanningEnv
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

DIM = 5

class ReplayBuffer:
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

    def add(self, states, action, reward, terminal, next_states):
        assert(type(states) is list)
        assert(type(next_states) is list)
        assert(len(states) == len(self.state_arrays))
        assert(len(next_states) == len(self.next_state_arrays))

        for i in range(0, len(states)):
            self.state_arrays[i][self.insert_index, ...] = states[i]
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.terminal[self.insert_index] = terminal
        for i in range(0, len(next_states)):
            self.next_state_arrays[i][self.insert_index, ...] = next_states[i]

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

        return state_arrays, actions, rewards, terminal, next_state_arrays

def create_cnn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            filters = 8,
            kernel_size = 4,
            strides = 1,
            activation = 'relu',
            padding = "same",
            input_shape = (2*DIM+1, 2*DIM+1, 1)
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size = 2,
            padding = "same"
        ),
        tf.keras.layers.Conv2D(
            filters = 16,
            kernel_size = 4,
            strides = 1,
            activation = 'relu',
            padding = "same",
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size = 2,
            padding = "same"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32)
    ])

def create_dnn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, input_dim = 2, activation = "relu"),
        tf.keras.layers.Dense(8, activation = "relu")
    ])

def create_nn():
    cnn = create_cnn()
    dnn = create_dnn()
        
    input = tf.keras.layers.concatenate([cnn.output, dnn.output])
    x = tf.keras.layers.Dense(16, activation = "relu")(input)
    output = tf.keras.layers.Dense(8, activation = "softmax")(x)

    return tf.keras.models.Model(inputs = [cnn.input, dnn.input], outputs=output)

def main():
    nn = create_nn()
    env = PathPlanningEnv("grid1.bmp", DIM)
    rb = ReplayBuffer(50, [(11, 11), 2])
    state = env.reset(np.array([10, 10]), np.array([3, 3]))
    for i in range(0, 500):
        print(state)
        prediction = nn.predict(state)
        action = np.argmax(prediction)
        next_state, reward, terminal = env.step(action)
        rb.add(state, action, reward, terminal, next_state)
        state = next_state

if __name__=='__main__':
    main()