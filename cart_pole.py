import copy
from collections import namedtuple
import numpy as np
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from gym.envs.classic_control import CartPoleEnv

DIM = 2

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

    def test(self, size):
        state_arrays = [None] * len(self.state_arrays)
        for i in range(0, len(state_arrays)):
            state_arrays[i] = np.random.choice(self.state_arrays[i], size)

        return state_arrays

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
    q_target_max = tf.reduce_max(q_target(next_states), axis=1)

    q_target_values = rewards + tf.multiply(q_target_max, 1.0 - tf.cast(terminals, tf.float32)) * gamma

    with tf.GradientTape() as tape:
        q_values = tf.reduce_sum(tf.multiply(q(states), actions_one_hot), axis=1)
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
    total_rewards = [None] * 25
    while True:
        state = env.reset(random=True)
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
                """ actions_one_hot = tf.keras.utils.to_categorical(actions, env.num_actions, dtype=np.float32)
                q_target_max = q_target.predict(next_states).max(axis=1)

                q_target_values = rewards + np.multiply(q_target_max, np.invert(terminals)) * gamma

                with tf.GradientTape() as tape:
                    q_values = tf.reduce_sum(tf.multiply(q(states), actions_one_hot), axis=1)
                    loss = tf.reduce_mean(tf.square(q_target_values - q_values))

                gradients = tape.gradient(loss, q.trainable_variables)
                q.optimizer.apply_gradients(zip(gradients, q.trainable_variables)) """

                train_step(env, q, q_target, gamma, states, actions, rewards, terminals, next_states)

                #print("Iteration {} complete with loss {}".format(iteration, loss))

            state = next_state

            if iteration % copy_interval == 0:
                q_target.set_weights(q.get_weights())

            if terminal:
                break
        
        total_rewards[episodes % 25] = total_reward
        episodes += 1

        r = 0.0
        c = 0
        for i in range(0, min(25, episodes)):
            r += total_rewards[i]
            c += 1
        print("Episode {}, episode reward: {}, average reward: {}".format(episodes, total_reward, r / c))

class CartPoleEnvWrapper:
    def __init__(self):
        self.env = CartPoleEnv()
        self.num_actions = 2

    def reset(self, random=True):
        state = self.env.reset()
        return [state]

    def step(self, action):
        state, reward, terminal, _ = self.env.step(action)
        return [state], reward, terminal

    def render(self):
        self.env.render()

def main():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(200, input_dim = 4, activation = "tanh"),
        tf.keras.layers.Dense(200, input_dim = 4, activation = "tanh"),
        tf.keras.layers.Dense(2, activation = "linear")
    ])
    model.compile(
        optimizer = tf.keras.optimizers.Adam(0.01)
    )
    env = CartPoleEnvWrapper()
    dqn(env, model, 0.99, 0.25, 200, 100000, 64, 25)

if __name__=='__main__':
    main()