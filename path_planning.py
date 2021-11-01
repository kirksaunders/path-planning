from ddqn import DDQN
from env import PathPlanningEnv
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

DIM = 4

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

def main():
    max_episode_steps = 50
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0015, max_episode_steps, 0.9995)
    exploration_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.5, max_episode_steps, 0.9995)
    beta = lambda it: min(1.0, 0.5 + it*0.00001)

    model = create_nn(learning_rate)
    env = PathPlanningEnv("grid_single_wall.bmp", DIM)
    agent = DDQN(env, model, 65536, 64, 0.7)
    agent.train(0.999, exploration_rate, max_episode_steps, 128, beta)

if __name__=='__main__':
    main()