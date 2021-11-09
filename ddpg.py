import numpy as np
import os

from tensorflow.python.eager.backprop import GradientTape
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from prioritized_replay_buffer import *
from replay_buffer import *

def state_to_tf_input(state):
    return [x.reshape((1, *(x.shape))) for x in state]

class DDPG:
    def __init__(self, env, actor, critic, replay_buffer):
        self.env = env
        self.actor = actor
        self.actor_target = tf.keras.models.clone_model(actor)
        self.actor_target.set_weights(actor.get_weights())
        self.critic = critic
        self.critic_target = tf.keras.models.clone_model(critic)
        self.critic_target.set_weights(critic.get_weights())
        self.replay_buffer = replay_buffer
        self.use_per = isinstance(self.replay_buffer, PrioritizedReplayBuffer)
        self.iterations = 0
        self.episodes = 0
    
    """ @tf.function
    def train_step_per(self, gamma, states, actions, rewards, terminals, next_states, weights):
        actions_one_hot = tf.one_hot(actions, self.env.num_actions)
        q_next_argmax = tf.argmax(self.q(next_states, training=True), axis=1)
        q_next_actions = tf.one_hot(q_next_argmax, self.env.num_actions)
        q_next_values = tf.reduce_sum(tf.multiply(self.q_target(next_states, training=True), q_next_actions), axis=1)

        q_target_values = rewards + tf.multiply(q_next_values, 1.0 - tf.cast(terminals, tf.float32)) * gamma

        with tf.GradientTape() as tape:
            q_values = tf.reduce_sum(tf.multiply(self.q(states, training=True), actions_one_hot), axis=1)
            td_error = q_target_values - q_values
            loss = tf.multiply(tf.reduce_mean(tf.square(td_error)), weights)

        gradients = tape.gradient(loss, self.q.trainable_variables)
        self.q.optimizer.apply_gradients(zip(gradients, self.q.trainable_variables))

        return tf.abs(td_error) """

    @tf.function
    def train_step(self, gamma, states, actions, rewards, terminals, next_states, weights):
        next_actions = self.actor_target(next_states, training=True)
        y = rewards + gamma * self.critic_target([next_states, next_actions], training=True)

        # Perform gradient descent on critic
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions], training=True)
            td_error = y - q_values
            loss = tf.reduce_mean(tf.square(td_error))

        gradients = tape.gradient(loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Perform gradient ascent on actor
        with tf.GradientTape() as tape:
            actions = self.actor(next_states, training=True)
            q_values = self.critic([next_states, actions], training=True)
            # Negative for ascent rather than descent
            loss = -tf.reduce_mean(q_values)

        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradient(zip(gradients, self.actor.trainable_variables))

    @tf.function
    def update_weights(target, main, tau):
        for (x, y) in zip(target.variables, main.variables):
            x.assign(tau * y + (1 - tau) * x)

    def train(self, gamma, action_noise, episode_step_limit, tau_actor, tau_critic, train_interval):
        total_rewards = [None] * 100
        while True:
            state = self.env.reset(random=True)
            total_reward = 0.0
            for t in range(0, episode_step_limit):
                self.iterations += 1

                action = self.actor(state_to_tf_input(state)) + action_noise(self.iterations)
                next_state, reward, terminal = self.env.step(action)
                total_reward += reward
                self.replay_buffer.add([state, action, reward, terminal, next_state])

                if self.replay_buffer.size >= self.replay_buffer.batch_size and self.iterations % train_interval == 0:
                    if self.use_per:
                        states, actions, rewards, terminals, next_states, weights, indices = self.replay_buffer.mini_batch()
                        priorities = self.train_step_per(gamma, states, actions, rewards, terminals, next_states, weights)
                        self.replay_buffer.update(indices, priorities.numpy())
                    else:
                        states, actions, rewards, terminals, next_states = self.replay_buffer.mini_batch()
                        self.train_step(gamma, states, actions, rewards, terminals, next_states)

                state = next_state

                # Move target networks in direction of main networks
                self.update_weights(self.actor_target, self.actor, tau_actor)
                self.update_weights(self.critic_target, self.critic, tau_critic)

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
                print("Episode {}, learning rate: {}, average reward: {}".format(
                    self.episodes, self.q.optimizer.learning_rate(self.iterations), r / c))

                #self.actor.save("results/ep{}_actor.h5".format(self.episodes))
                #self.critic.save("results/ep{}_critic.h5".format(self.episodes))

                """ state = self.env.reset(random=True)
                for t in range(0, episode_step_limit):
                    action = np.argmax(self.q(state_to_tf_input(state), training=False))
                    next_state, reward, terminal = self.env.step(action)
                    if terminal:
                        break
                    state = next_state
                self.env.display() """
