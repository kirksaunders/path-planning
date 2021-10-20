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

def state_to_tf_input(state):
    return [x.reshape((1, *(x.shape))) for x in state]

def model_input_shape(nn):
    shapes = [None] * len(nn.input)
    for i in range(0, len(nn.input)):
        _, *rest = nn.input[i].shape
        shapes[i] = tuple(rest)

    return shapes

def dqn(env, q, gamma, epsilon, episode_step_limit, replay_size, batch_size, copy_interval):
    replay_buffer = ReplayBuffer(replay_size, model_input_shape(q))
    q_target = tf.keras.model.clone_model(q)
    q_target.set_weights(q.get_weights())

    iteration = 0
    while True:
        state = env.reset(random=True)
        for t in range(0, episode_step_limit):
            iteration += 1
            if np.random.random() > epsilon:
                action = q.predict(state_to_tf_input(state)).argmax()
            else:
                action = np.random.choice(env.num_actions)

            next_state, reward, terminal = env.step(action)
            replay_buffer.add(state, action, reward, terminal, next_state)

            states, actions, rewards, terminals, next_states = replay_buffer.mini_batch(min(iteration, batch_size))
            actions_one_hot = tf.keras.utils.to_categorical(actions, env.num_actions, dtype=np.float32)
            q_target_max = q_target.predict(next_states).max(axis=1)

            q_target = rewards + q_target_max.multiply(terminals.invert()) * gamma

            with tf.GradientTape() as tape:
                q_values = tf.reduce_sum(tf.multiply(q.predict(states), actions_one_hot), axis=1)
                loss = tf.reduce_mean(tf.square(q_target - q_values))

            gradients = tape.gradient(loss, q.trainable_variables)
            q.optimizer.apply_gradients(zip(gradients, q.trainable_variables))

            print("Iteration {} complete with loss {}".format(iteration, loss))

            state = next_state

            if iteration % copy_interval == 0:
                q_target.set_weights(q.get_weights())

            if terminal:
                break

def main():
    model = create_nn()
    env = PathPlanningEnv("grid1.bmp", DIM)
    dqn(env, model, 0.999, 0.1, 100, 128, 32, 100)

if __name__=='__main__':
    main()