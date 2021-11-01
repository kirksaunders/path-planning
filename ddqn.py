import numpy as np
from replaybuffer import ReplayBuffer
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

def state_to_tf_input(state):
    return [x.reshape((1, *(x.shape))) for x in state]

""" def model_input_shape(nn):
    input = nn.input
    if not (type(input) is list):
        input = [input]

    shapes = [None] * len(input)
    for i in range(0, len(input)):
        _, *rest = input[i].shape
        shapes[i] = tuple(rest)

    return shapes """

# Training step for dqn (not double)
""" @tf.function
def train_step(env, q, q_target, gamma, states, actions, rewards, terminals, next_states):
    actions_one_hot = tf.one_hot(actions, env.num_actions)
    q_target_max = tf.reduce_max(q_target(next_states, training=True), axis=1)

    q_target_values = rewards + tf.multiply(q_target_max, 1.0 - tf.cast(terminals, tf.float32)) * gamma

    with tf.GradientTape() as tape:
        q_values = tf.reduce_sum(tf.multiply(q(states, training=True), actions_one_hot), axis=1)
        loss = tf.reduce_mean(tf.square(q_target_values - q_values))

    gradients = tape.gradient(loss, q.trainable_variables)
    q.optimizer.apply_gradients(zip(gradients, q.trainable_variables)) """

class DDQN:
    def __init__(self, env, q, replay_size):
        self.env = env
        self.q = q
        self.q_target = tf.keras.models.clone_model(q)
        self.q_target.set_weights(q.get_weights())
        self.replay_buffer = ReplayBuffer(replay_size)
        self.iterations = 0
        self.episodes = 0
    
    @tf.function
    def train_step(self, gamma, states, actions, rewards, terminals, next_states, priorities):
        actions_one_hot = tf.one_hot(actions, self.env.num_actions)
        q_next_argmax = tf.argmax(self.q(next_states, training=True), axis=1)
        q_next_actions = tf.one_hot(q_next_argmax, self.env.num_actions)
        q_next_values = tf.reduce_sum(tf.multiply(self.q_target(next_states, training=True), q_next_actions), axis=1)

        q_target_values = rewards + tf.multiply(q_next_values, 1.0 - tf.cast(terminals, tf.float32)) * gamma

        with tf.GradientTape() as tape:
            q_values = tf.reduce_sum(tf.multiply(self.q(states, training=True), actions_one_hot), axis=1)
            td_error = q_target_values - q_values
            loss = tf.reduce_mean(tf.square(td_error))

        gradients = tape.gradient(loss, self.q.trainable_variables)
        self.q.optimizer.apply_gradients(zip(gradients, self.q.trainable_variables))

        return tf.abs(td_error)

    def train(self, gamma, epsilon, episode_step_limit, batch_size, copy_interval):
        total_rewards = [None] * 100
        while True:
            state = self.env.reset(random=True)
            total_reward = 0.0
            for t in range(0, episode_step_limit):
                self.iterations += 1
                if np.random.random() > epsilon(self.iterations):
                    action = np.argmax(self.q(state_to_tf_input(state)))
                else:
                    action = np.random.choice(self.env.num_actions)

                next_state, reward, terminal = self.env.step(action)
                total_reward += reward
                self.replay_buffer.add((state, action, reward, terminal, next_state))

                if self.replay_buffer.size >= batch_size:
                    states, actions, rewards, terminals, next_states, priorities, indices = self.replay_buffer.mini_batch(batch_size)
                    updated_priorities = self.train_step(gamma, states, actions, rewards, terminals, next_states, priorities)
                    self.replay_buffer.update(indices, updated_priorities)

                state = next_state

                if self.iterations % copy_interval == 0:
                    self.q_target.set_weights(self.q.get_weights())

                if terminal:
                    break

            total_rewards[self.episodes % 100] = total_reward
            self.episodes += 1

            if self.episodes % 100 == 0:
                r = 0.0
                c = 0
                for i in range(0, min(100, self.episodes)):
                    r += total_rewards[i]
                    c += 1
                print("Episode {}, learning rate: {}, epsilon: {}, episode reward: {}, average reward: {}".format(
                    self.episodes, self.q.optimizer.learning_rate(self.iterations), epsilon(self.iterations), total_reward, r / c))

                state = self.env.reset(random=True)
                for t in range(0, episode_step_limit):
                    action = np.argmax(self.q(state_to_tf_input(state), training=False))
                    next_state, reward, terminal = self.env.step(action)
                    if terminal:
                        break
                    state = next_state
                self.env.draw()
