# MeshDQN
Welcome to MeshDQN, the official repository.

MeshDQN uses double deep Q learning to iteratively coarsen meshes in computational fluid dynamics.
For more details about the method, please read the [paper]().

## Running MeshDQN

MeshDQN is supported on [docker](https://www.docker.com/).

Running the docker for MeshDQN image can be done simply as:
`docker run -v /[path]/[to]/[parent]/[directory]/:/home/fenics/ -ti quay.io/fenicsproject/stable:meshdqn`

The config files for various runs are found in the `configs/` directory, with accompanying airfoil files found in the `xdmf_files/` directory.
Data for the results in the paper are provided here.

The training script for running MeshDQN, with relevant training/DQN functions is `airfoil_dqn.py`.
This script is ready to run for the ys930 airfoil.
This can be done with `python3 airfoil_dqn.py`.

After running, the `training_results` directory will be created for all training results, with `ys930_results` directory created inside for this specific run.
All of the relevant information for each run is stored in the results directory, including the config, models, losses, action selection history, etc.
It is recommended to modify the `PREFIX` variable for different runs.
If a run stops, or needs to be stopped, changing `RESTART` to `True` will restart the run from the most recent previous saved state.
Note: if the run dies unexpectedly during saving, some of the training data may be corrupted and not be able to be processed.

While MeshDQN is training, progress can be monitored by running the provided `plot_reward.py` and `analyze_actions.py` scripts.
`plot_reward.py` will plot moving averages of the reward which helps see if it is improving over time.
`analyze_actions.py` plots the model loss and action selection histogram.

During or after training, the model can be deployed on the current airfoil mesh with `deploy_dqn.py`, which creates the `deployed` directory inside the results directory.
`complete_traj` will run a full simulation after each vertex removal and is used to generate the figures in the paper.
It makes deployment significantly slower, and can be set to false if you are interested only in the final result.
`plot_traj` plots the mesh after each vertex removal.
`end_plots` will plot only the initial and final meshes.
`use_best` will select the episode with highest reward from training and use those actions instead of the DQN actions.
`RESTART` is used if the run you would like to deploy from has been restarted.
`CONFIRM` is used to run an additional confirmation deployment to verify results.

After deployment, the results figures from the paper can be generated with `analyze_benchmark.py`.


## Evaluation

If you use MeshDQN, please cite as: [AVAILABLE TOMORROW]


