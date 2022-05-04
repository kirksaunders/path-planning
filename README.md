# Path Planning through DRL
This repo contains code accompanying my research in path planning using deep reinforcement techniques.

## Dependencies
Some dependencies are needed to run the code in this repo:
- Python 3
- Numpy
- Tensorflow
- Tkinter (for visualizing agent ability)
- PIL (for rendering results to PNGs)
- Tensorboard (for seeing training logs/graphs)
- A decent PC (otherwise training will be very slow)

## Usage
Find the various grid maps in the grids directory. Choose one or create your own.

Ensure your current working directory is the root of this repo.

The python script accepts command line arguments. Run the command with the `-h` flag to see usage.

To start training:
```
python src/path_planning_continuous.py -g grids/*your grid here* -d log_save_dir/
```

Every 5 episodes you should receive a summary of the reward over those episodes in the console. You should also see a window that displays the results of the previous episode. Simply ctrl+c to stop.

In the results directory, the networks are saved after each 5 episodes. Some other data is also saved (currently just the rewards).

To resume training with an existing network:
```
python src/path_planning_continuous.py -g grids/*your grid here* -m existing_saved_model_path -d log_save_dir/
```
where `existing_saved_model_path` is for example `some_results/ep50` (do not put `_actor` or a file extension).

To evaluate performance of a trained agent:
```
python src/path_planning_continuous.py -g grids/*your grid here* -e -m existing_saved_model_path
```

A window should appear showing the environment. Left click anywhere to set the start position and right click anywhere to set the goal position. After the agent finishes, the result will also be saved to results/out.png.