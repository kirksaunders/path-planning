import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from ..memory.prioritized_replay_buffer import *
from ..memory.replay_buffer import *

def state_to_tf_input(state):
    return [x.reshape((1, *(x.shape))) for x in state]

class DDPG:
    """
    Agent implementing the Deep Deterministic Policy Gradient algorithm.
    See paper (https://arxiv.org/abs/1509.02971).
    """
    
    def __init__(self, env, actor, critic, replay_buffer):
        """
        Initialize agent with given environment, actor+critic networks, and replay buffer.
        """
        
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

    @tf.function
    def _train_step(self, gamma, states, actions, rewards, terminals, next_states):
        next_actions = self.actor_target(next_states, training=True)
        q_next_values = self.critic_target([next_states, next_actions], training=True)
        y = rewards + gamma * tf.multiply(q_next_values, 1.0 - tf.cast(terminals, tf.float32))

        # Perform gradient descent on critic
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions], training=True)
            td_error = y - q_values
            critic_loss = tf.reduce_mean(tf.square(td_error))

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Perform gradient ascent on actor
        with tf.GradientTape() as tape:
            new_actions = self.actor(states, training=True)
            q_values = self.critic([states, new_actions], training=True)
            # Negative for ascent rather than descent
            actor_loss = -tf.reduce_mean(q_values)

        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        return critic_loss, actor_loss

    @tf.function
    def _train_step_per(self, gamma, states, actions, rewards, terminals, next_states, weights):
        next_actions = self.actor_target(next_states, training=True)
        q_next_values = self.critic_target([next_states, next_actions], training=True)
        y = rewards + gamma * tf.multiply(q_next_values, 1.0 - tf.cast(terminals, tf.float32))

        # Perform gradient descent on critic
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions], training=True)
            td_error = y - q_values
            critic_loss = tf.reduce_mean(tf.multiply(tf.square(td_error), weights))

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Perform gradient ascent on actor
        with tf.GradientTape() as tape:
            new_actions = self.actor(states, training=True)
            q_values = self.critic([states, new_actions], training=True)
            # Negative for ascent rather than descent
            actor_loss = -tf.reduce_mean(q_values)

        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        return tf.abs(td_error), critic_loss, actor_loss

    @tf.function
    def _update_weights(self, target, main, tau):
        for (x, y) in zip(target.variables, main.variables):
            x.assign(tau * y + (1 - tau) * x)

    def train(self, gamma, action_noise, episode_step_limit, tau_actor, tau_critic, train_interval, report_interval, log_dir):
        """
        Train agent with parameters:
            gamma: discount factor
            action_noise: noise function to add to action from actor network
            episode_step_limit: number of steps to end each episode at
            tau_actor: amount to move target actor network towards learned network each step
            tau_actor: amount to move target critic network towards learned network each step
            train_interval: number of iterations that need to pass for each training step
            report_interval: number of episodes that need to pass for each progress message and network save
            log_dir: directory to save logs to
        """

        log_writer = tf.summary.create_file_writer(str(log_dir / "logs"))
        critic_loss_avg = tf.keras.metrics.Mean("critic_loss",  dtype=tf.float32)
        actor_loss_avg = tf.keras.metrics.Mean("actor_loss",  dtype=tf.float32)

        total_rewards = [None] * report_interval
        while True:
            state = self.env.reset(start=np.array([70.0, 70.0]), goal=np.array([5.0, 5.0]))
            total_reward = 0.0
            critic_loss_avg.reset_states()
            actor_loss_avg.reset_states()
            for t in range(0, episode_step_limit):
                self.iterations += 1

                action = self.actor(state_to_tf_input(state)).numpy() + action_noise(self.iterations)
                action = action.reshape(action.shape[1:])
                norm = np.linalg.norm(action)
                if norm > 1.0:
                    action /= norm
                #action = np.clip(action, -1.0, 1.0)
                next_state, reward, terminal = self.env.step(action)
                total_reward += reward
                self.replay_buffer.add([state, action, reward, terminal, next_state])

                if self.replay_buffer.size >= self.replay_buffer.batch_size and self.iterations % train_interval == 0:
                    if self.use_per:
                        states, actions, rewards, terminals, next_states, weights, indices = self.replay_buffer.mini_batch()
                        priorities, critic_loss, actor_loss = self._train_step_per(gamma, states, actions, rewards, terminals, next_states, weights)
                        self.replay_buffer.update(indices, priorities.numpy())
                    else:
                        states, actions, rewards, terminals, next_states = self.replay_buffer.mini_batch()
                        critic_loss, actor_loss = self._train_step(gamma, states, actions, rewards, terminals, next_states)

                    critic_loss_avg(critic_loss)
                    actor_loss_avg(actor_loss)

                state = next_state

                # Move target networks in direction of main networks
                self._update_weights(self.actor_target, self.actor, tau_actor)
                self._update_weights(self.critic_target, self.critic, tau_critic)

                if terminal:
                    break

            total_rewards[self.episodes % report_interval] = total_reward
            self.episodes += 1

            with log_writer.as_default():
                tf.summary.scalar("critic_loss", critic_loss_avg.result(), step=self.episodes)
                tf.summary.scalar("actor_loss", actor_loss_avg.result(), step=self.episodes)
                tf.summary.scalar("reward", total_reward, step=self.episodes)

            if self.episodes % report_interval == 0:
                r = 0.0
                c = 0
                for i in range(0, report_interval):
                    r += total_rewards[i]
                    c += 1
                print("Episode {}, average reward (of last {}): {}".format(
                    self.episodes, report_interval, r / c))

                # Save networks to log_dir
                self.actor.save(log_dir / "models/ep{}_actor.h5".format(self.episodes))
                self.critic.save(log_dir / "models/ep{}_critic.h5".format(self.episodes))

                # Display and print result _without_ action noise
                state = self.env.reset(start=np.array([70.0, 70.0]), goal=np.array([5.0, 5.0]))
                total_reward = 0.0
                for t in range(0, episode_step_limit):
                    self.iterations += 1

                    action = self.actor(state_to_tf_input(state)).numpy()
                    action = action.reshape(action.shape[1:])
                    next_state, reward, terminal = self.env.step(action)
                    total_reward += reward

                    state = next_state

                    if terminal:
                        break

                print("Test reward: {}".format(total_reward))

                # Display state of most recent episode
                self.env.display()
