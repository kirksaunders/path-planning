from ddqn import DDQN
from ddqn import state_to_tf_input
from env import PathPlanningEnv
import numpy as np
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tkinter as tk
from time import sleep

DIM = 5

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

def create_nn(lr):
    cnn = create_cnn()
    dnn = create_dnn()
        
    input = tf.keras.layers.concatenate([cnn.output, dnn.output])
    x = tf.keras.layers.Dense(32, activation = "relu")(input)
    output = tf.keras.layers.Dense(8, activation = "linear")(x)

    model = tf.keras.models.Model(inputs = [cnn.input, dnn.input], outputs=output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr)
    )

    return model

def train():
    tk_root = tk.Tk()

    max_episode_steps = 75
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0015, max_episode_steps, 0.9995)
    exploration_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.5, max_episode_steps, 0.9995)
    beta = lambda it: min(1.0, 0.5 + it*0.00001)

    model = create_nn(learning_rate)
    env = PathPlanningEnv("grid_single_wall.bmp", DIM, tk_root)
    agent = DDQN(env, model, 65536, 64, 0.7)
    agent.train(0.999, exploration_rate, max_episode_steps, 128, beta)

def evaluate(model_file):
    tk_root = tk.Tk()

    max_episode_steps = 100

    model = tf.keras.models.load_model(model_file)

    start = np.array([0, 0])
    end = np.array([5, 5])

    def run():
        nonlocal env, start, end

        total_reward = 0
        state = env.reset(start, end)
        for t in range(0, max_episode_steps):
            action = np.argmax(model(state_to_tf_input(state)))
            state, reward, terminal = env.step(action)
            total_reward += reward
            env.display()
            if terminal:
                break
            sleep(0.1)

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

    env = PathPlanningEnv("grid_single_wall.bmp", DIM, tk_root, on_click_left, on_click_right)
    env.display()

    tk_root.mainloop()


if __name__=='__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == "evaluate":
        evaluate(sys.argv[2])
    else:
        train()