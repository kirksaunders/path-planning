import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from ..memory.prioritized_replay_buffer import *
from ..memory.replay_buffer import *

def state_to_tf_input(state):
    return [x.reshape((1, *(x.shape))) for x in state]

class TD3:
    """
    Agent implementing the Twin Delayed Deep Deterministic policy gradient algorithm.
    See paper (https://arxiv.org/abs/1802.09477).
    WARNING: Not well-tested and not ready for use.
    """

    def __init__(self, env, actor, critic, replay_buffer):
        self.env = env

        # One actor network
        self.actor = actor

        # One target actor network
        self.actor_target = tf.keras.models.clone_model(actor)
        self.actor_target.set_weights(actor.get_weights())

        # Two critic networks
        critic2 = tf.keras.models.clone_model(critic)
        critic2.optimizer = tf.keras.optimizers.Adam(critic.optimizer.learning_rate)
        #critic2.set_weights(critic.get_weights())
        self.critics = [critic, critic2]

        # Two target critic networks
        critic_target1 = tf.keras.models.clone_model(critic)
        critic_target1.set_weights(critic.get_weights())
        critic_target2 = tf.keras.models.clone_model(critic2)
        critic_target2.set_weights(critic2.get_weights())
        self.critic_targets = [critic_target1, critic_target2]

        self.replay_buffer = replay_buffer
        self.use_per = isinstance(self.replay_buffer, PrioritizedReplayBuffer)
        self.iterations = 0
        self.episodes = 0

        self.rng = np.random.default_rng()

    @tf.function
    def train_step(self, gamma, states, actions, rewards, terminals, next_states):
        next_actions = self.actor_target(next_states, training=True)
        q_next_values = self.critic_target([next_states, next_actions], training=True)
        y = rewards + gamma * tf.multiply(q_next_values, 1.0 - tf.cast(terminals, tf.float32))

        # Perform gradient descent on critic
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions], training=True)
            td_error = y - q_values
            loss = tf.reduce_mean(tf.square(td_error))

        gradients = tape.gradient(loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Perform gradient ascent on actor
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            q_values = self.critic([states, actions], training=True)
            # Negative for ascent rather than descent
            loss = -tf.reduce_mean(q_values)

        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

    @tf.function
    def train_step_per(self, gamma, states, actions, rewards, terminals, next_states, weights, noise, update_actor):
        next_actions = tf.squeeze(self.actor_target(next_states, training=True)) + tf.squeeze(noise)
        q_next_values1 = self.critic_targets[0]([next_states, next_actions], training=True)
        q_next_values2 = self.critic_targets[1]([next_states, next_actions], training=True)
        q_next_values = tf.minimum(q_next_values1, q_next_values2)
        y = rewards + gamma * tf.multiply(q_next_values, 1.0 - tf.cast(terminals, tf.float32))

        # Perform gradient descent on critic 1
        with tf.GradientTape() as tape:
            q_values1 = self.critics[0]([states, actions], training=True)
            td_error1 = y - q_values1
            loss = tf.reduce_mean(tf.multiply(tf.square(td_error1), weights))

        gradients = tape.gradient(loss, self.critics[0].trainable_variables)
        self.critics[0].optimizer.apply_gradients(zip(gradients, self.critics[0].trainable_variables))

        # Perform gradient descent on critic 2
        with tf.GradientTape() as tape:
            q_values2 = self.critics[1]([states, actions], training=True)
            td_error2 = y - q_values2
            loss = tf.reduce_mean(tf.multiply(tf.square(td_error2), weights))

        gradients = tape.gradient(loss, self.critics[1].trainable_variables)
        self.critics[1].optimizer.apply_gradients(zip(gradients, self.critics[1].trainable_variables))

        # Take average of td error for PER
        td_error = (td_error1 + td_error2) * 0.5

        if update_actor:
            # Perform gradient ascent on actor
            with tf.GradientTape() as tape:
                actions = tf.squeeze(self.actor(states, training=True))
                # Always use first critic in this step (as per paper)
                q_values = self.critics[0]([states, actions], training=True)
                # Negative for ascent rather than descent
                loss = -tf.reduce_mean(q_values)

            gradients = tape.gradient(loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        return tf.squeeze(tf.abs(td_error))

    @tf.function
    def update_weights(self, target, main, tau):
        for (x, y) in zip(target.variables, main.variables):
            x.assign(tau * y + (1 - tau) * x)

    def train(self, gamma, action_noise, episode_step_limit, tau_actor, tau_critic, train_interval, policy_update_interval):
        total_rewards = [None] * 40
        while True:
            state = self.env.reset(random=True)
            total_reward = 0.0
            for t in range(0, episode_step_limit):
                self.iterations += 1

                action = self.actor(state_to_tf_input(state)).numpy() + action_noise(self.iterations)
                action = np.squeeze(action)
                next_state, reward, terminal = self.env.step(action)
                total_reward += reward
                self.replay_buffer.add([state, action, reward, terminal, next_state])

                if self.replay_buffer.size >= self.replay_buffer.batch_size and self.iterations % train_interval == 0:
                    noise = np.reshape(self.rng.random(self.replay_buffer.batch_size * 2, dtype=np.float32)*0.5 - 0.25, (self.replay_buffer.batch_size, 2))
                    if self.use_per:
                        states, actions, rewards, terminals, next_states, weights, indices = self.replay_buffer.mini_batch()
                        priorities = self.train_step_per(gamma, states, actions, rewards, terminals, next_states, weights, noise, self.iterations % policy_update_interval == 0)
                        self.replay_buffer.update(indices, priorities.numpy())
                    else:
                        states, actions, rewards, terminals, next_states = self.replay_buffer.mini_batch()
                        self.train_step(gamma, states, actions, rewards, terminals, next_states)

                    if self.iterations % policy_update_interval == 0:
                        # Move target networks in direction of main networks
                        self.update_weights(self.actor_target, self.actor, tau_actor)
                        self.update_weights(self.critic_targets[0], self.critics[0], tau_critic)
                        self.update_weights(self.critic_targets[1], self.critics[1], tau_critic)

                state = next_state

                if terminal:
                    break

            total_rewards[self.episodes % 40] = total_reward
            self.episodes += 1

            if self.episodes % 5 == 0:
                r = 0.0
                c = 0
                for i in range(0, min(40, self.episodes)):
                    r += total_rewards[i]
                    c += 1
                print("Episode {}, learning rate: {}, average reward: {}".format(
                    self.episodes, 0, r / c))

                self.actor.save("results/ep{}_actor.h5".format(self.episodes))
                # I suppose only save first critic (for now)
                self.critics[0].save("results/ep{}_critic.h5".format(self.episodes))

                state = self.env.reset(random=True)
                for t in range(0, episode_step_limit):
                    action = self.actor(state_to_tf_input(state), training=False)
                    next_state, reward, terminal = self.env.step(action)
                    if terminal:
                        break
                    state = next_state
                self.env.display()
