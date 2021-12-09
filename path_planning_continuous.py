import numpy as np
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tkinter as tk

from ddpg import *
from env_continuous import *
from td3 import *
from uniform_replay_buffer import *
from proportional_replay_buffer import *

DIM = 5

def create_cnn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            filters = 4,
            kernel_size = 4,
            strides = 1,
            activation = "relu",
            input_shape = (2*DIM+1, 2*DIM+1, 1)
        ),
        tf.keras.layers.AveragePooling2D(
            pool_size = 2,
        ),
        tf.keras.layers.Conv2D(
            filters = 8,
            kernel_size = 2,
            strides = 1,
            activation = "relu",
        ),
        tf.keras.layers.AveragePooling2D(
            pool_size = 2,
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = "relu"),
        #tf.keras.layers.Dense(8, activation = "relu")
    ])

def create_dnn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, input_dim = 2, activation = "relu"),
        #tf.keras.layers.Dense(8, activation = "relu")
    ])

def create_actor_model(lr):
    cnn = create_cnn()
    dnn = create_dnn()

    initializer = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    input = tf.keras.layers.concatenate([cnn.output, dnn.output])
    x = tf.keras.layers.Dense(128, activation = "relu")(input)
    x = tf.keras.layers.Dense(64, activation = "relu")(input)
    x = tf.keras.layers.Dense(2, activation = "tanh", kernel_initializer=initializer)(x)
    output = x * 1.0 # Allow displacement distance of 2 in each dimension each step

    model = tf.keras.models.Model(inputs = [cnn.input, dnn.input], outputs=output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr, clipnorm=1.0)
    )

    return model

def create_critic_model(lr):
    cnn = create_cnn()
    dnn = create_dnn()

    initializer = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    x = tf.keras.layers.concatenate([cnn.output, dnn.output])
    state_output = tf.keras.layers.Dense(64, activation = "relu")(x)
    
    action_input = tf.keras.layers.Input(shape=(2,))
    action_output = tf.keras.layers.Dense(16, activation = "relu")(action_input)

    x = tf.keras.layers.concatenate([state_output, action_output])
    x = tf.keras.layers.Dense(128, activation = "relu")(x)
    x = tf.keras.layers.Dense(64, activation = "relu")(x)
    output = tf.keras.layers.Dense(1, activation = "linear", kernel_initializer=initializer)(x)

    model = tf.keras.models.Model([[cnn.input, dnn.input], action_input], output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr)
    )

    return model

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

def train(model_file = None):
    tk_root = tk.Tk()

    max_episode_steps = 400
    batch_size = 16

    learning_rate_actor = tf.keras.optimizers.schedules.ExponentialDecay(0.00015, max_episode_steps, 0.995)
    learning_rate_critic = tf.keras.optimizers.schedules.ExponentialDecay(0.0015, max_episode_steps, 0.995)
    action_noise = OUActionNoise(np.zeros(2), 0.2 * np.ones(2))
    #rng = np.random.default_rng()
    #asd = OUActionNoise(np.zeros(2), 0.2 * np.ones(2))
    #action_noise = lambda it: rng.random() - 0.5 if it < 10000 else asd(it)
    tau = 0.001
    beta = lambda it: min(1.0, 0.5 + it*0.00001)

    if model_file == None:
        actor = create_actor_model(learning_rate_actor)
        critic = create_critic_model(learning_rate_critic)
    else:
        actor = tf.keras.models.load_model(model_file + "_actor.h5")
        critic = tf.keras.models.load_model(model_file + "_critic.h5")

    #rb = UniformReplayBuffer(1000000, batch_size)
    rb = ProportionalReplayBuffer(1000000, batch_size, 0.6, beta)

    env = ContinuousPathPlanningEnv("grid4.bmp", DIM, tk_root)
    agent = DDPG(env, actor, critic, rb)
    #agent = TD3(env, actor, critic, rb)

    agent.train(0.99, action_noise, max_episode_steps, tau, tau, 1)
    #agent.train(0.99, action_noise, max_episode_steps, tau, tau, 1, 2)

def evaluate(model_file):
    tk_root = tk.Tk()

    max_episode_steps = 500

    actor = tf.keras.models.load_model(model_file + "_actor.h5")

    # Grid 3
    #start = np.array([90, 90])
    #end = np.array([10, 10])

    # Grid 4
    #start = np.array([60, 65])
    #end = np.array([10, 10])

    # Grid 5
    start = np.array([65, 65])
    end = np.array([10, 10])

    input = 0

    def run():
        nonlocal env, start, end, input

        input += 1
        input_copy = input

        total_reward = 0
        state = env.reset(start, end)
        for t in range(0, max_episode_steps):
            if input != input_copy:
                break
            action = actor(state_to_tf_input(state), training=False)
            state, reward, terminal = env.step(action)
            total_reward += reward
            env.display()
            if terminal:
                break

        env.draw_img()

    def on_click_left(event):
        nonlocal start

        x = event.x / env.draw_size
        y = event.y / env.draw_size

        start = np.array([x, y])
        run()

    def on_click_right(event):
        nonlocal end

        x = event.x / env.draw_size
        y = event.y / env.draw_size

        end = np.array([x, y])
        run()

    env = ContinuousPathPlanningEnv("grid_empty_large.bmp", DIM, tk_root, on_click_left, on_click_right)
    env.display()

    run()

    tk_root.mainloop()

if __name__=='__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == "evaluate":
        evaluate(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "resume":
        train(sys.argv[2])
    else:
        train()