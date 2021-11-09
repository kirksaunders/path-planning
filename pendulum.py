from gym.envs.classic_control import PendulumEnv
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from ddpg import *
from uniform_replay_buffer import *

def create_actor_model(lr):
    initializer = tf.keras.initializers.VarianceScaling(2.0)

    input = tf.keras.layers.Dense(16, input_dim = 3, activation = "relu", kernel_initializer=initializer)
    x = tf.keras.layers.Dense(1, activation = "tanh")(input)
    output = x * 2.0

    model = tf.keras.models.Model(input, output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr, clipnorm=1.0)
    )

def create_critic_model(lr):
    initializer = tf.keras.initializers.VarianceScaling(2.0)

    state_input = tf.keras.layers.Dense(16, input_dim = 3, activation = "relu", kernel_initializer=initializer)
    state_output = tf.keras.layers.Dense(1, activation = "tanh")(state_input)

    action_inout = tf.keras.layers.Dense(16, input_dim = 1, activation = "relu", kernel_initializer=initializer)

    x = tf.keras.layers.concatenate([state_output, action_inout])
    x = tf.keras.layers.Dense(16, activation = "relu", kernel_initializer=initializer)
    output = tf.keras.layers.Dense(1, activation = "linear")

    model = tf.keras.models.Model([state_input, action_inout], output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr, clipnorm=1.0)
    )

class PendulumEnvWrapper:
    def __init__(self):
        self.env = PendulumEnv()

    def reset(self, random=True):
        state = self.env.reset()
        return [state]

    def step(self, action):
        state, reward, terminal, _ = self.env.step(action)
        return [state], reward, terminal

    def render(self):
        self.env.render()

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self, its=None):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


def train():
    max_episode_steps = 200
    batch_size = 64

    #learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0015, max_episode_steps, 0.9995)
    action_noise = OUActionNoise(np.zeros(1), 0.2 * np.ones(1))
    #beta = lambda it: min(1.0, 0.5 + it*0.00001)

    actor = create_actor_model(0.001)
    critic = create_critic_model(0.002)

    rb = UniformReplayBuffer(1000000, batch_size)
    #rb = ProportionalReplayBuffer(1000000, batch_size, 0.6, beta)

    env = PendulumEnvWrapper()
    agent = DDPG(env, actor, critic, rb)
    agent.train(0.999, action_noise, max_episode_steps, 0.005, 0.005, 1)

if __name__ == "__main__":
    train()