import numpy as np
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tkinter as tk
from time import sleep

from ddqn import DDQN, state_to_tf_input
from env import PathPlanningEnv
from uniform_replay_buffer import *
from proportional_replay_buffer import *

DIM = 5

def create_cnn():
    initializer = tf.keras.initializers.VarianceScaling(2.0)

    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            filters = 4,
            kernel_size = 4,
            strides = 1,
            activation = "relu",
            kernel_initializer = initializer,
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
            kernel_initializer = initializer,
        ),
        tf.keras.layers.AveragePooling2D(
            pool_size = 2,
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = "relu", kernel_initializer=initializer),
        #tf.keras.layers.Dense(8, activation = "relu")
    ])

def create_dnn():
    initializer = tf.keras.initializers.VarianceScaling(2.0)

    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, input_dim = 2, activation = "relu", kernel_initializer=initializer),
        #tf.keras.layers.Dense(8, activation = "relu")
    ])

def create_nn(lr):
    cnn = create_cnn()
    dnn = create_dnn()

    initializer = tf.keras.initializers.VarianceScaling(2.0)
        
    input = tf.keras.layers.concatenate([cnn.output, dnn.output])
    x = tf.keras.layers.Dense(32, activation = "relu", kernel_initializer=initializer)(input)
    #x = tf.keras.layers.Dense(16, activation = "linear")(x)
    output = tf.keras.layers.Dense(8, activation = "linear")(x)

    model = tf.keras.models.Model(inputs = [cnn.input, dnn.input], outputs=output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr, clipnorm=1.0)
    )

    return model

def train(model_file = None):
    tk_root = tk.Tk()

    max_episode_steps = 200
    batch_size = 64

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0015, max_episode_steps, 0.9995)
    exploration_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.5, max_episode_steps, 0.9995)
    beta = lambda it: min(1.0, 0.5 + it*0.00001)

    if model_file == None:
        model = create_nn(learning_rate)
    else:
        model = tf.keras.models.load_model(model_file)

    #rb = UniformReplayBuffer(1000000, batch_size)
    rb = ProportionalReplayBuffer(1000000, batch_size, 0.6, beta)

    env = PathPlanningEnv("grid2.bmp", DIM, tk_root)
    agent = DDQN(env, model, rb, batch_size)
    agent.train(0.999, exploration_rate, max_episode_steps, 250, beta)

def evaluate(model_file):
    tk_root = tk.Tk()

    max_episode_steps = 100

    model = tf.keras.models.load_model(model_file)

    start = np.array([0, 0])
    end = np.array([5, 5])
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
            action = np.argmax(model(state_to_tf_input(state), training=False))
            state, reward, terminal = env.step(action)
            total_reward += reward
            env.display()
            if terminal:
                break
            sleep(0.025)

    def on_click_left(event):
        nonlocal start

        x = event.x // env.draw_size
        y = event.y // env.draw_size

        start = np.array([x, y])
        run()

    def on_click_right(event):
        nonlocal end

        x = event.x // env.draw_size
        y = event.y // env.draw_size

        end = np.array([x, y])
        run()

    env = PathPlanningEnv("grid2.bmp", DIM, tk_root, on_click_left, on_click_right)
    env.display()

    tk_root.mainloop()


if __name__=='__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == "evaluate":
        evaluate(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "resume":
        train(sys.argv[2])
    else:
        train()