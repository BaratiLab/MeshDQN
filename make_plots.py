from gym import Env, spaces
from flow_solver import FlowSolver
import numpy as np
import yaml

from dolfin import *
set_log_active(False)

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from scipy.spatial import Delaunay

from tqdm import tqdm
import time

from torch import nn
from torch import optim

# TODO: These are misnomers
from airfoilgcnn import AirfoilGCNN, NodeRemovalNet
#from airfoilgcnn_training import PredictionNet
from Env2DAirfoil import Env2DAirfoil
from ClosestEnv2DAirfoil import ClosestEnv2DAirfoil
from ClosestSmoothingEnv2DAirfoil import ClosestSmoothingEnv2DAirfoil

from itertools import count

import random

import torch
from torch.nn import functional as F

from collections import namedtuple

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import os

from shapely.geometry import Polygon, Point
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if(torch.cuda.is_available()):
    print("USING GPU")
    device = torch.device('cuda:0')
else:
    print("USING CPU")
    device = torch.device('cpu')

def plot_mesh(mesh, name="mesh", title=None, actions=None, vertex_coords=None):
    coords = mesh.coordinates()
    edges = []
    for c in mesh.cells():
        edges.append([c[0], c[1]])
        edges.append([c[0], c[2]])
        edges.append([c[1], c[2]])

    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(coords[:,0], coords[:,1], color='k', s=5, zorder=1)
    for e in edges:
        ax.plot([coords[e[0]][0], coords[e[1]][0]],
                [coords[e[0]][1], coords[e[1]][1]],
                color="#888888", lw=0.5, zorder=0)

    actual_actions = []
    if(actions is not None):
        for a in actions:
            #a = env.mesh_map[a]
            actual_actions.append(a)
            #ax.scatter(coords[a][0], coords[a][1], color='r', s=10)
        print("ACTUAL ACTIONS TAKEN: {}".format(actual_actions))

    if(vertex_coords is not None):
        for v in vertex_coords:
            ax.scatter(v[0], v[1], color='r', s=10)

    if(title):
        ax.set_title(title, fontsize=14)

    plt.savefig("./{}.png".format(name), bbox_inches="tight")


def plot_state(env, title="{}", filename="example_state.pdf"):
    state = env.get_state()
    mesh = env.flow_solver.mesh
    closest = env.n_closest
    #print(closest)

    edges = []
    coords = mesh.coordinates()
    for c in mesh.cells():
        edges.append([c[0], c[1]])
        edges.append([c[0], c[2]])
        edges.append([c[1], c[2]])

    fig, ax = plt.subplots(figsize=(10,5))

    colors = np.array(['r', 'k'])
    removable = np.array(env.flow_solver.removable).astype(int)
    ax.scatter(coords[:,0], coords[:,1], color=colors[removable], s=6, zorder=1)
    for e in edges:
        ax.plot([coords[e[0]][0], coords[e[1]][0]],
                [coords[e[0]][1], coords[e[1]][1]],
                color="#888888", lw=0.75, zorder=0)

    for selected_coord in env.coord_map.values():
        ax.scatter(coords[selected_coord][0], coords[selected_coord][1], color='b', s=6)

    edges = state.edge_index
    for e in range(edges.shape[1]):
        p1 = coords[env.coord_map[int(edges[0][e])]]
        p2 = coords[env.coord_map[int(edges[1][e])]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='b', lw=0.75)


    custom_handles = [
            Line2D([0],[0], color='r', marker='o', lw=0, markersize=3),
            Line2D([0],[0], color='k', marker='o', lw=0.5, markersize=3),
            Line2D([0],[0], color='b', marker='o', lw=0.5, markersize=3),
    ]
    ax.legend(custom_handles, ['Not Removable', 'Removable - Not in State', 'Removable - In State'],
              bbox_to_anchor=[0.05,0.03,0.93,0], ncol=3, fontsize=12)


    ax.set_title(title.format(env.N_CLOSEST), fontsize=18, y=0.975)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_axis_off()
    #plt.savefig("./mesh_plots/example_state.png", bbox_inches='tight')
    plt.savefig("./mesh_plots/{}.png".format(filename), bbox_inches='tight')


if __name__ == '__main__':
    PREFIX = "NONE"
    #OLD_PREFIX = "test_closest_fine_10step_ag11_"
    OLD_PREFIX = "deep_wide_300_do_nothing_ag11_18"
    OLD_PREFIX = "ag11_20_do_nothing_tuning_time_and_drag_"
    OLD_PREFIX = "ys930_please_work"

    OLD_PREFIX = 'ag11_18_'
    OLD_PREFIX = 'ys930_16_'

    #file_name = "ag11_18_final_"
    file_name = "ys930_16_final_"
    title_diff = ""
    
    # Set up environment
    #with open("./configs/ag11.yaml", 'r') as stream:
    with open("./configs/ys930.yaml", 'r') as stream:
    #with open("./configs/s102s.yaml", 'r') as stream:
    #with open("./configs/cylinder.yaml", 'r') as stream:
    #with open("./configs/square.yaml", 'r') as stream:
        flow_config = yaml.safe_load(stream)
    flow_config['agent_params']['solver_steps'] = 5
    print(flow_config['agent_params']['do_nothing'])
    env = ClosestSmoothingEnv2DAirfoil(flow_config)
    
    # Hold on to ground truth values
    flow_config['agent_params']['gt_drag'] = env.gt_drag
    flow_config['agent_params']['gt_time'] = env.gt_time
    flow_config['agent_params']['u'] = env.u.copy(deepcopy=True)
    flow_config['agent_params']['p'] = env.p.copy(deepcopy=True)
    n_actions = env.action_space.n
    
    # Set up for DQN
    policy_net_1 = NodeRemovalNet(n_actions, conv_width=256, topk=0.1).to(device).float()
    
    # Prime and load policy net
    policy_net_1.set_num_nodes(env.initial_num_node)
    #policy_net_1.load_state_dict(
    #        torch.load("./trained_dqns/{}policy_net_1.pt".format(OLD_PREFIX),
    #        map_location=device
    #))
    #print("SUCCESSFULLY LOADED POLICY NET")

    plot_state(env, title="{} Closest Vertices to Airfoil", filename="initial_state")
    for i in tqdm(range(300)):
        _ = env.step(200)
    plot_state(env, title="Next {} Closest Vertices to Airfoil", filename="do_nothing_state")
    
    #plot_mesh(env.flow_solver.mesh, "./mesh_plots/original_{}_mesh".format(file_name),
    #          "Original {} Mesh".format(title_diff), actions, vertex_coords)
    
