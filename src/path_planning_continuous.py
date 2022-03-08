import numpy as np
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tkinter as tk

from drl.agents.ddpg import *
from drl.environments.continuous_path_planning import *
from drl.memory.uniform_replay_buffer import *
from drl.memory.proportional_replay_buffer import *

# Amount of the map that the agents sees. DIM=5 is 11x11 view. (2*DIM+1)x(2*DIM+1) in general.
DIM = 5

# Number of frames of input to the network
NUM_FRAMES = 1

def create_cnn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            filters = 4,
            kernel_size = 4,
            strides = 1,
            activation = "relu",
            data_format = "channels_first",
            input_shape = (NUM_FRAMES, 2*DIM+1, 2*DIM+1)
        ),
        tf.keras.layers.AveragePooling2D(
            pool_size = 2,
            data_format = "channels_first"
        ),
        tf.keras.layers.Conv2D(
            filters = 8,
            kernel_size = 2,
            strides = 1,
            data_format = "channels_first",
            activation = "relu",
        ),
        tf.keras.layers.AveragePooling2D(
            pool_size = 2,
            data_format = "channels_first"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = "relu"),
    ])

""" def create_cnn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((NUM_FRAMES, 2*DIM+1, 2*DIM+1)),
        tf.keras.layers.Reshape((NUM_FRAMES, 1, 2*DIM+1, 2*DIM+1)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
            filters = 4,
            kernel_size = 4,
            strides = 1,
            activation = "relu",
            data_format = "channels_first",
        )),
        tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(
            pool_size = 2,
            data_format = "channels_first"
        )),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
            filters = 8,
            kernel_size = 2,
            strides = 1,
            data_format = "channels_first",
            activation = "relu",
        )),
        tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(
            pool_size = 2,
            data_format = "channels_first"
        )),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.GRU(64),
        #tf.keras.layers.Dense(64, activation = "relu"),
    ]) """

def create_dnn():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(NUM_FRAMES, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation = "relu"),
    ])

""" def create_dnn():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(NUM_FRAMES, 2)),
        tf.keras.layers.GRU(16),
        #tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation = "relu"),
    ]) """

def create_actor_model(lr):
    cnn = create_cnn()
    dnn = create_dnn()

    initializer = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    input = tf.keras.layers.concatenate([cnn.output, dnn.output])
    x = tf.keras.layers.Dense(128, activation = "relu")(input)
    x = tf.keras.layers.Dense(64, activation = "relu")(input)
    output = tf.keras.layers.Dense(2, activation = "tanh", kernel_initializer=initializer)(x)

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

# Class taken from keras examples. (https://keras.io/examples/rl/ddpg_pendulum/)
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

def train(grid, model_file = None):
    tk_root = tk.Tk()

    max_episode_steps = 400
    batch_size = 16
    train_interval = 1
    report_interval = 5

    learning_rate_actor = tf.keras.optimizers.schedules.ExponentialDecay(0.00015, max_episode_steps, 0.995)
    learning_rate_critic = tf.keras.optimizers.schedules.ExponentialDecay(0.0015, max_episode_steps, 0.995)
    action_noise = OUActionNoise(np.zeros(2), 0.2 * np.ones(2))
    tau = 0.001
    beta = lambda it: min(1.0, 0.5 + it*0.00001)

    if model_file == None:
        actor = create_actor_model(learning_rate_actor)
        critic = create_critic_model(learning_rate_critic)
    else:
        actor = tf.keras.models.load_model(model_file + "_actor.h5")
        critic = tf.keras.models.load_model(model_file + "_critic.h5")

    # Plot network structure
    """from tensorflow.keras.utils import plot_model
    plot_model(actor, to_file="actor_model.png", show_shapes=True, show_layer_names=False)
    plot_model(critic, to_file="critic_model.png", show_shapes=True, show_layer_names=False)"""

    #rb = UniformReplayBuffer(1000000, batch_size)
    rb = ProportionalReplayBuffer(1000000, batch_size, 0.6, beta)

    env = ContinuousPathPlanningEnv(grid, DIM, NUM_FRAMES, tk_root)
    agent = DDPG(env, actor, critic, rb)

    agent.train(0.99, action_noise, max_episode_steps, tau, tau, train_interval, report_interval)

def evaluate(grid, model_file):
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
    start = np.array([65, 65], dtype=np.float32)
    end = np.array([10, 10], dtype=np.float32)

    input = 0

    def run():
        nonlocal env, start, end, input

        input += 1
        input_copy = input

        total_reward = 0
        state = env.reset(start, end, random=False)
        env.display()
        for t in range(0, max_episode_steps):
            if input != input_copy:
                break
            action = actor(state_to_tf_input(state), training=False)
            state, reward, terminal = env.step(action)
            total_reward += reward
            env.display(only_latest=True)
            if terminal:
                break

        env.draw_img()

    def on_click_left(event):
        nonlocal start

        x = event.x / env.draw_size
        y = event.y / env.draw_size

        start = np.array([x, y], dtype=np.float32)
        run()

    def on_click_right(event):
        nonlocal end

        x = event.x / env.draw_size
        y = event.y / env.draw_size

        end = np.array([x, y], dtype=np.float32)
        run()

    env = ContinuousPathPlanningEnv(grid, DIM, NUM_FRAMES, tk_root, on_click_left, on_click_right)
    env.display()

    run()

    tk_root.mainloop()

if __name__=='__main__':
    if len(sys.argv) >= 4 and sys.argv[2] == "evaluate":
        evaluate(sys.argv[1], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[2] == "resume":
        train(sys.argv[1], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[2] == "train":
        train(sys.argv[1])