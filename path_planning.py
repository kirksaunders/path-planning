import copy
from collections import namedtuple
from env import PathPlanningEnv
import numpy as np
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

DIM = 4

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

class ReplayBufferNew:
    def __init__(self, capacity):
        self.data = [None] * capacity
        self.capacity = capacity
        self.size = 0
        self.index = 0
        self.tuple = namedtuple("Experience", ["states", "actions", "rewards", "terminals", "next_states"])

    def add(self, experience):
        self.data[self.index] = experience
        self.index += 1
        if self.index >= self.capacity:
            self.index = 0

        if self.size < self.capacity:
            self.size += 1

    def mini_batch(self, size):
        data = random.sample(self.data[0:self.size], size)
        data = self.tuple(*zip(*data))

        states = [None] * len(data[0][0])
        next_states = [None] * len(data[0][0])
        for i in range(0, len(states)):
            states[i] = np.asarray([x[i] for x in data[0]], dtype=np.float32)
            next_states[i] = np.asarray([x[i] for x in data[4]], dtype=np.float32)

        return states, np.asarray(data[1], dtype=np.int32), np.asarray(data[2], dtype=np.float32), np.asarray(data[3], dtype=bool), next_states

def create_cnn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            filters = 4,
            kernel_size = 3,
            strides = 1,
            activation = "relu",
            padding = "same",
            input_shape = (2*DIM+1, 2*DIM+1, 1)
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size = 2,
            padding = "same"
        ),
        tf.keras.layers.Conv2D(
            filters = 8,
            kernel_size = 3,
            strides = 1,
            activation = "relu",
            padding = "same",
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size = 2,
            padding = "same"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation = "tanh"),
        #tf.keras.layers.Dense(8, activation = "relu")
    ])

def create_dnn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, input_dim = 2, activation = "tanh"),
        #tf.keras.layers.Dense(8, activation = "relu")
    ])

def create_nn():
    cnn = create_cnn()
    dnn = create_dnn()
        
    input = tf.keras.layers.concatenate([cnn.output, dnn.output])
    x = tf.keras.layers.Dense(32, activation = "relu")(input)
    output = tf.keras.layers.Dense(8, activation = "linear")(x)

    model = tf.keras.models.Model(inputs = [cnn.input, dnn.input], outputs=output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(0.0015)
    )

    return model

def state_to_tf_input(state):
    return [x.reshape((1, *(x.shape))) for x in state]

def model_input_shape(nn):
    input = nn.input
    if not (type(input) is list):
        input = [input]

    shapes = [None] * len(input)
    for i in range(0, len(input)):
        _, *rest = input[i].shape
        shapes[i] = tuple(rest)

    return shapes

@tf.function
def train_step(env, q, q_target, gamma, states, actions, rewards, terminals, next_states):
    actions_one_hot = tf.one_hot(actions, env.num_actions)
    q_target_max = tf.reduce_max(q_target(next_states, training=True), axis=1)

    q_target_values = rewards + tf.multiply(q_target_max, 1.0 - tf.cast(terminals, tf.float32)) * gamma

    with tf.GradientTape() as tape:
        q_values = tf.reduce_sum(tf.multiply(q(states, training=True), actions_one_hot), axis=1)
        loss = tf.reduce_mean(tf.square(q_target_values - q_values))

    gradients = tape.gradient(loss, q.trainable_variables)
    q.optimizer.apply_gradients(zip(gradients, q.trainable_variables))

def dqn(env, q, gamma, epsilon, episode_step_limit, replay_size, batch_size, copy_interval):
    #replay_buffer = ReplayBuffer(replay_size, model_input_shape(q))
    replay_buffer = ReplayBufferNew(replay_size)
    q_target = tf.keras.models.clone_model(q)
    q_target.set_weights(q.get_weights())

    iteration = 0
    episodes = 0
    total_rewards = [None] * 100
    while True:
        epsilon = epsilon*0.9995
        q.optimizer.learning_rate = q.optimizer.learning_rate*0.9995
        state = env.reset(random=True)
        """ if episodes % 3 == 0:
            state = env.reset(np.array([13, 1]), np.array([16, 12]))
        elif episodes % 3 == 1:
            state = env.reset(np.array([4, 3]), np.array([11, 17]))
        elif episodes % 3 == 2:
            state = env.reset(np.array([19, 10]), np.array([1, 10])) """
        total_reward = 0.0
        for t in range(0, episode_step_limit):
            iteration += 1
            if np.random.random() > epsilon:
                action = np.argmax(q(state_to_tf_input(state)))
            else:
                action = np.random.choice(env.num_actions)

            next_state, reward, terminal = env.step(action)
            total_reward += reward
            #replay_buffer.add(state, action, reward, terminal, next_state)
            replay_buffer.add((state, action, reward, terminal, next_state))

            if replay_buffer.size >= batch_size:
                states, actions, rewards, terminals, next_states = replay_buffer.mini_batch(batch_size)
                train_step(env, q, q_target, gamma, states, actions, rewards, terminals, next_states)

            state = next_state

            if iteration % copy_interval == 0:
                q_target.set_weights(q.get_weights())

            if terminal:
                break
        
        total_rewards[episodes % 100] = total_reward
        episodes += 1

        if episodes % 100 == 0:
            r = 0.0
            c = 0
            for i in range(0, min(100, episodes)):
                r += total_rewards[i]
                c += 1
            print("Episode {}, learning rate: {}, epsilon: {}, episode reward: {}, average reward: {}".format(episodes, q.optimizer.learning_rate.numpy(), epsilon, total_reward, r / c))

            #state = env.reset(random=True)
            state = env.reset(
                np.array([np.random.choice(20), np.random.choice(8)]),
                np.array([np.random.choice(20), np.random.choice(8) + 11])
            )
            """ if episodes % 3 == 0:
                state = env.reset(np.array([13, 1]), np.array([16, 12]))
            elif episodes % 3 == 1:
                state = env.reset(np.array([4, 3]), np.array([11, 17]))
            elif episodes % 3 == 2:
                state = env.reset(np.array([19, 10]), np.array([1, 10])) """
            for t in range(0, episode_step_limit):
                action = np.argmax(q(state_to_tf_input(state), training=False))
                next_state, reward, terminal = env.step(action)
                if terminal:
                    break
                state = next_state
            env.display()

def main():
    model = create_nn()
    env = PathPlanningEnv("grid_single_wall.bmp", DIM)
    dqn(env, model, 0.999, 0.5, 50, 65536, 64, 32)

if __name__=='__main__':
    main()