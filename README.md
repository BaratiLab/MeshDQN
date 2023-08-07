# MeshDQN
Welcome to MeshDQN, the official repository.

MeshDQN uses double deep Q learning to iteratively coarsen meshes in computational fluid dynamics.
For more details about the method, please read the [paper](https://pubs.aip.org/aip/adv/article/13/1/015026/2871176/Mesh-deep-Q-network-A-deep-reinforcement-learning).

## Running MeshDQN

MeshDQN is supported on [docker](https://www.docker.com/). In order to run MeshDQN an account must be created there first.
The Docker image can be downloaded from the command line with `docker pull cooplo/meshdqn:latest`.
Running the image is then: `docker run -v /[path]/[to]/[parent]/[directory]/:/home/fenics/ -ti cooplo/meshdqn:latest`

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

If you use MeshDQN, please cite as: 
```
@article{10.1063/5.0138039,
    author = {Lorsung, Cooper and Barati Farimani, Amir},
    title = "{Mesh deep Q network: A deep reinforcement learning framework for improving meshes in computational fluid dynamics}",
    journal = {AIP Advances},
    volume = {13},
    number = {1},
    year = {2023},
    month = {01},
    abstract = "{Meshing is a critical, but user-intensive process necessary for stable and accurate simulations in computational fluid dynamics (CFD). Mesh generation is often a bottleneck in CFD pipelines. Adaptive meshing techniques allow the mesh to be updated automatically to produce an accurate solution for the problem at hand. Existing classical techniques for adaptive meshing require either additional functionality out of solvers, many training simulations, or both. Current machine learning techniques often require substantial computational cost for training data generation, and are restricted in scope to the training data flow regime. Mesh Deep Q Network (MeshDQN) is developed as a general purpose deep reinforcement learning framework to iteratively coarsen meshes while preserving target property calculation. A graph neural network based deep Q network is used to select mesh vertices for removal and solution interpolation is used to bypass expensive simulations at each step in the improvement process. MeshDQN requires a single simulation prior to mesh coarsening, while making no assumptions about flow regime, mesh type, or solver, only requiring the ability to modify meshes directly in a CFD pipeline. MeshDQN successfully improves meshes for two 2D airfoils.}",
    issn = {2158-3226},
    doi = {10.1063/5.0138039},
    url = {https://doi.org/10.1063/5.0138039},
    note = {015026},
    eprint = {https://pubs.aip.org/aip/adv/article-pdf/doi/10.1063/5.0138039/16698492/015026\_1\_online.pdf},
}
```


