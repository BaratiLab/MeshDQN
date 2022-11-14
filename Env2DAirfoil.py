import sys
sys.path.append(sys.path[2][:-18])

from gym import Env, spaces
from flow_solver import FlowSolver
from probes import DragProbe, LiftProbe, PenetratedDragProbe
import numpy as np
import yaml

from dolfin import *
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from airfoilgcnn import AirfoilGCNN

from scipy.spatial import Delaunay

from tqdm import tqdm
import time

import torch
from torch import nn

from torch_geometric.data import Data
import os
from shapely.geometry import Polygon, Point

#from multiprocessing import pool
from joblib import Parallel, delayed
#from pathos.multiprocessing import ProcessingPool as Pool
import ray

if(torch.cuda.is_available()):
    print("USING GPU")
    device = torch.device("cuda:0")
else:
    print("USING CPU")
    device = torch.device("cpu")
device = torch.device("cpu")

#@ray.remote
class Env2DAirfoil(Env):
    """
        Environment to optimize mesh around a 2D airfoil
    """

    def __init__(self, config):
        #super(ParallelMultiSnapshotEnv2DAirfoil, self).__init__()

        # Define flow solver
        self.flow_solver = FlowSolver(**config['flow_config'])

        # Keep track of mesh vertices
        self.coordinate_list = list(range(len(self.flow_solver.mesh.coordinates())))
        self.initial_num_node = len(self.coordinate_list)
        self.removable = np.argwhere(self.flow_solver.removable)[:,0] # Can now check if mesh broke from this

        # Map from removable vertex index to original index
        self.mesh_map = {idx: self.coordinate_list.index(rem) for idx, rem in enumerate(self.removable)}
        removable = np.array(self.flow_solver.removable, dtype=int)
        colors = np.array(['b', 'r'])

        # Trying to focus only on closest nodes to airfoil
        self.N_CLOSEST = config['agent_params']['N_closest']
        self.TIME_REWARD = config['agent_params']['time_reward']

        # Define action space
        self.action_space = spaces.Discrete(self.N_CLOSEST) # Only removable vertices can be selected

        self.solver_steps = config['agent_params']['solver_steps']
        self.episodes = config['agent_params']['episodes']
        self.timesteps = config['agent_params']['timesteps']
        self.threshold = config['agent_params']['threshold']
        self.NEGATIVE_REWARD = -1.

        # For holding onto later
        self.removed_coordinates = []
        self.do_nothing_offset = 0

        # Agent Parameters
        self.gt_drag = np.array(config['agent_params']['gt_drag'])
        self.gt_time = np.array(config['agent_params']['gt_time'])
        self.u = config['agent_params']['u']
        self.p = config['agent_params']['p']
        self.original_u = config['agent_params']['u']
        self.original_p = config['agent_params']['p']
        self.save_steps = config['agent_params']['save_steps']
        self.goal_vertices = config['agent_params']['goal_vertices']
        self.plot_dir = config['agent_params']['plot_dir']

        if(not(isinstance(self.u, int))):
            self.u = [u.copy(deepcopy=True) for u in config['agent_params']['u']]
            self.p = [p.copy(deepcopy=True) for p in config['agent_params']['p']]
            self.original_u = [u.copy(deepcopy=True) for u in config['agent_params']['u']]
            self.original_p = [p.copy(deepcopy=True) for p in config['agent_params']['p']]

        self.POLYGON = False
        self.out_of_vertices = False
        self.reset()


    def reset(self):

        # Turn ground truths into a list
        if(self.gt_drag.shape == ()):
            self.gt_drag = np.array([self.gt_drag])
        if(self.gt_time.shape == ()):
            self.gt_time = np.array([self.gt_time])

        # Run Rimulation if necessary
        if((self.gt_drag[0] == -1) and (self.gt_time[0] == -1)):
            self.gt_drag, self.gt_lift, self.original_u, self.original_p, self.p, self.u = ([] for i in range(6))
            print("CALCULATING INITIAL VALUE...")
            start = time.time()
            for i in tqdm(range(self.solver_steps)):
                u, p, drag, lift = self.flow_solver.evolve()

                if((i+1)%self.save_steps == 0):
                    #print("\n\nSAVING AT STEP: {}\n\n".format(i+1))
                    self.gt_drag.append(drag)
                    self.gt_lift.append(lift)
                    self.original_u.append(u.copy(deepcopy=True))
                    self.original_p.append(p.copy(deepcopy=True))
                    self.u.append(u.copy(deepcopy=True))
                    self.p.append(p.copy(deepcopy=True))
        else: # Load saved values
            self.original_u, self.original_p, self.p, self.u = ([] for i in range(4))
            self.velocities = np.load("/home/fenics/drl_projects/MeshDQN/parallel_training/" + self.plot_dir + \
                                      "/snapshots/velocities.npy")
            self.pressures = np.load("/home/fenics/drl_projects/MeshDQN/parallel_training/" + self.plot_dir + \
                                     "/snapshots/pressures.npy")
            save_us = np.load("/home/fenics/drl_projects/MeshDQN/parallel_training/" + self.plot_dir + \
                              "/snapshots/save_velocities.npy")
            save_ps = np.load("/home/fenics/drl_projects/MeshDQN/parallel_training/" + self.plot_dir + \
                              "/snapshots/save_pressures.npy")
            for i in range(int(np.ceil(self.solver_steps/self.save_steps))):
                # This will almost certainly break down with any change in underlying fenics data structure
                # Creat new u function
                U_new = VectorFunctionSpace(self.flow_solver.mesh, 'Lagrange', 2)
                u_func = Function(U_new)#, degree=2)
                u_func.vector().set_local(save_us[i].flatten())
                self.original_u.append(u_func.copy(deepcopy=True))
                self.u.append(u_func.copy(deepcopy=True))

                # Create new p function
                P_new = FunctionSpace(self.flow_solver.mesh, 'Lagrange', 1)
                p_func = Function(P_new)#, degree=1)
                p_func.vector().set_local(save_ps[i])
                self.original_p.append(p_func.copy(deepcopy=True))
                self.p.append(p_func.copy(deepcopy=True))

                del u_func
                del p_func

        
        # Get and save velocities
        self._calculate_velocities()
        self._calculate_pressures()

        # For keeping track of progress
        self.steps = 0
        self.num_episodes = 0
        self.terminal = False
        self._get_distance_lookup()


    def return_vals(self):
        return self.gt_drag, self.gt_time


    def plot_state(self, title="{}", filename="initial_state.pdf"):
        state = self.get_state()
        mesh = self.flow_solver.mesh
        closest = self.n_closest
        #print(closest)
        
        edges = []
        coords = mesh.coordinates()
        for c in mesh.cells():
            edges.append([c[0], c[1]])
            edges.append([c[0], c[2]])
            edges.append([c[1], c[2]])
        
        fig, ax = plt.subplots(figsize=(10,5))
        
        colors = np.array(['r', 'k'])
        removable = np.array(self.flow_solver.removable).astype(int)
        ax.scatter(coords[:,0], coords[:,1], color=colors[removable], s=6, zorder=1)
        for e in edges:
            ax.plot([coords[e[0]][0], coords[e[1]][0]],
                    [coords[e[0]][1], coords[e[1]][1]],
                    color="#888888", lw=0.75, zorder=0)
            
        for selected_coord in self.coord_map.values():
            ax.scatter(coords[selected_coord][0], coords[selected_coord][1], color='b', s=6)
        
        edges = state.edge_index
        for e in range(edges.shape[1]):
            p1 = coords[self.coord_map[int(edges[0][e])]]
            p2 = coords[self.coord_map[int(edges[1][e])]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='b', lw=0.75)
        
                
        custom_handles = [
                Line2D([0],[0], color='r', marker='o', lw=0, markersize=3),
                Line2D([0],[0], color='k', marker='o', lw=0.5, markersize=3),
                Line2D([0],[0], color='b', marker='o', lw=0.5, markersize=3),
        ]
        ax.legend(custom_handles, ['Not Removable', 'Removable - Not in State', 'Removable - In State'    ],
                  bbox_to_anchor=[0.05,0.03,0.93,0], ncol=3, fontsize=12)
        
        
        ax.set_title(title.format(self.N_CLOSEST), fontsize=18, y=0.975)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_axis_off()
        plt.savefig("./{}/{}.png".format(self.plot_dir, filename), bbox_inches='tight')


    def _get_distance_lookup(self):
        removable = np.array(self.flow_solver.removable, dtype=int)
        coords = self.flow_solver.mesh.coordinates()
        if(not self.POLYGON):
            not_removable = np.argwhere(-(np.array(self.flow_solver.removable)-1))[:,0]
            boundary_coords = coords[not_removable]
            airfoil_coords = boundary_coords[
                    np.logical_and((boundary_coords[:,0] > -0.5) ,
                    np.logical_and((boundary_coords[:,0] < 3),
                    np.logical_and((boundary_coords[:,1] > -0.5),
                    (boundary_coords[:,1] < 0.5))))
            ]
            self.polygon = Polygon(airfoil_coords)
            self.POLYGON = True

        # Precompute distances for later lookup 
        #self.distance_lookup = {}
        #for idx, coord in enumerate(coords[self.removable]):
        #    self.distance_lookup[idx] = self.polygon.distance(Point(coord))
        self.distance_lookup = []
        for idx, coord in enumerate(coords[self.removable]):
            self.distance_lookup.append(self.polygon.distance(Point(coord)))


    def get_state(self):
        # Turn current mesh into PyG data object for AirfoilGCNN
        edge_index, edge_attr = [], []

        # calculate N closest point to the airofil
        self._n_closest()

        # Only retain edge if its the N-closest 
        append_times, check_times, lookup_times = [], [], []

        # Need to remove this somehow for mesh smoothing... reintroduce vertex index?
        coord_map_vals = np.array(list(self.coord_map.values())).astype(int)

        # Prime loop
        mesh_cells = self.flow_solver.mesh.cells()
        good_idxs = np.argwhere(np.all(np.isin(mesh_cells, coord_map_vals), axis=1))
        coordinates = self.flow_solver.mesh.coordinates()
        for idx in good_idxs[:,0]:
            # Get ID of selected cell vertices in full mesh
            id1 = self.inv_coord_map[mesh_cells[idx][0]]
            id2 = self.inv_coord_map[mesh_cells[idx][1]]
            id3 = self.inv_coord_map[mesh_cells[idx][2]]

            # Get coordinatesof mesh cells
            c1 = coordinates[mesh_cells[idx]][0]
            c2 = coordinates[mesh_cells[idx]][1]
            c3 = coordinates[mesh_cells[idx]][2]

            # Calculate distance between pairs of points
            edge_attr.append(np.linalg.norm(c1-c2))
            edge_attr.append(np.linalg.norm(c1-c3))
            edge_attr.append(np.linalg.norm(c2-c3))

            # Create edges between points
            edge_index.append([id1, id2])
            edge_index.append([id1, id3])
            edge_index.append([id2, id3])

        # Stack and create data object
        edge_index = torch.LongTensor(edge_index).T

        x = torch.zeros((self.N_CLOSEST, 3*self.velocities.shape[0] + 2), dtype=torch.float)
        x[:,:2] = torch.from_numpy(self.flow_solver.mesh.coordinates()[self.n_closest])
        x[:,2:2*self.velocities.shape[0]+2] = torch.from_numpy(self.velocities[:,self.n_closest,:].reshape(self.N_CLOSEST,-1))
        x[:,2*self.velocities.shape[0]+2:] = torch.from_numpy(self.pressures[:,self.n_closest][:,:,0].T)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)


    def _n_closest(self):
        # Keep track of mesh vertices
        self.coordinate_list = list(range(len(self.flow_solver.mesh.coordinates())))
        self.removable = np.argwhere(self.flow_solver.removable)[:,0] # Can now check if mesh broke from this

        # Map from removable vertex index to original index
        vec_lookup = np.vectorize(lambda x: self.coordinate_list.index(x))
        self.mesh_map = dict(zip(range(len(self.removable)), vec_lookup(self.removable)))
        coords = self.flow_solver.mesh.coordinates()

        # Get distances to the airfoil
        self._get_distance_lookup()
        dist_idxs = np.argsort(self.distance_lookup)
        
        # Create new map between N-closest and original indices
        self.n_closest = dist_idxs[self.do_nothing_offset:self.N_CLOSEST + self.do_nothing_offset]
        if(len(self.n_closest) < self.N_CLOSEST):
            print("OUT OF VERTICES")
            self.out_of_vertices = True
        vec_map = np.vectorize(lambda x: self.mesh_map[x])
        mapping = vec_map(self.n_closest)
        self.coord_map = dict(zip(range(len(self.n_closest)), mapping))
        self.inv_coord_map = dict(zip(mapping, range(len(self.n_closest))))


    def step(self, action):
        '''
            Take a step by removing a vertex, calculating reward and determining
            if the episode has finished.

            TODO: Need optimal mesh criteria.
                  Need to determine if mesh is broken/solver will diverge without running
                    or by running every once in a while.
        '''
        step_time = time.time()
        broken = False

        if(action == self.action_space.n): # No removal - shift [0, N] closest to [1, N+1] closest
            self.do_nothing_offset += 1
            removed = 0
        else:
            # If vertex removal breaks simulation, big penalty, end episode
            removed = self._remove_vertex(action)

        state = self.get_state()
        if(self.out_of_vertices):
            print("OUT OF VERTICES")
            removed = 2
        
        if(removed == 0):    # Node was successfully removed
            # TODO: Catch when simulation diverges. End episode there.
            #       If drag calculation is outside some threshold, also end episode?
            start = time.time()
            rew, broken, self.terminal = self.calculate_reward()

            if(self.terminal):
                self.rew = 0.5*self.NEGATIVE_REWARD
                #print(np.abs(np.abs(self.gt_drag - self.new_drags)/self.gt_drag) > self.threshold)
                #print(np.abs(np.abs(self.gt_drag - self.new_drags)/self.gt_drag))
                #print("NEW DRAGS: {}".format(self.new_drags))
                print("ACCURACY THRESHOLD REACHED")

            if(broken):
                rew = self.NEGATIVE_REWARD
                self.terminal = True
            # If neither above conditions, save mesh...
        elif(removed == 1):  # Selected node has already been removed
            rew = self.NEGATIVE_REWARD
        elif(removed == 2):  # Node removal broke mesh
            rew = self.NEGATIVE_REWARD
            self.terminal = True
            broken = True

        #print("REWARD: {}".format(rew))
        self.steps += 1
        if(self.steps >= self.timesteps):
            self.terminal = True
            self.episodes += 1

        if(rew == np.nan):  # I'm not sure what's causing this
            rew = self.NEGATIVE_REWARD
        elif(isinstance(rew, torch.Tensor) and rew.isnan()):
            rew = self.NEGATIVE_REWARD

        return state, rew, self.terminal, {}


    def calculate_reward(self):
        '''
            Use AirfoilGCNN to calculate reward. No longer necessary to run simulation for
            each timestep.

            This does not check if mesh is broken or not. 
        '''

        try:
            self.new_drags, self.new_lifts = [], []
            for u, p in zip(self.u, self.p):
                new_drag = self.flow_solver.drag_probe.sample(u, p)
                new_lift = self.flow_solver.lift_probe.sample(u, p)
                self.new_drags.append(new_drag)
                self.new_lifts.append(new_lift)
        except:
            print("\n\nSAMPLING BROKE\n\n")
            return self.NEGATIVE_REWARD, True, True

        self.new_drags = np.array(self.new_drags)
        self.new_lifts = np.array(self.new_lifts)

        #print("NEW DRAGS: {}".format(self.new_drags))
        #print("GT  DRAGS: {}".format(self.gt_drag))
        #raise

        # Drag reward
        drag_factor = -2*np.log(0.5)/self.threshold
        #drag_factor = -4*np.log(0.5)/self.threshold
        #drag_factor = 5000
        error_val = np.linalg.norm(np.abs(self.gt_drag - self.new_drags)/np.abs(self.gt_drag))
        drag_reward = 2*np.exp(-drag_factor*error_val) - 1

        # Time reward
        time_reward = (self.initial_num_node - len(self.coordinate_list)) * self.TIME_REWARD

        # Accuracy threshold
        acc_thresh = any(np.abs(np.abs(self.gt_drag - self.new_drags)/self.gt_drag) > self.threshold)

        # Vertex Threshold
        vert_thresh = len(self.flow_solver.mesh.coordinates()) < self.goal_vertices * self.initial_num_node
        if(vert_thresh):
            print("\nMAXIMUM REMOVALS REACHED\n")

        if(drag_reward == np.nan):
            print(self.get_state())
            print("\n\nDRAG IS NAN!\n\n")
            raise
        return drag_reward+time_reward, False, acc_thresh or vert_thresh
               #any(np.abs(np.abs(self.gt_drag - self.new_drags)/self.gt_drag) > self.threshold)


    def set_plot_dir(self, plot_dir):
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(plot_dir+"/snapshots", exist_ok=True)

        # Save numpy format of pressure and velocity
        np.save(self.plot_dir + "/snapshots/velocities.npy", self.velocities)
        np.save(self.plot_dir + "/snapshots/pressures.npy", self.pressures)

        # Save primitive information from pressure and velocity
        save_us, save_ps = [], []
        for i in range(len(self.original_u)):
            save_us.append(self.original_u[i].vector().get_local())
            save_ps.append(self.original_p[i].vector().get_local())

        #print("\n\nSAVE PS SHAPE: {}\n\n".format(np.array(save_ps).shape))
        np.save(self.plot_dir + "/snapshots/save_velocities.npy", save_us)
        np.save(self.plot_dir + "/snapshots/save_pressures.npy", save_ps)


    def _remove_vertex(self, selected_coord=None):
        # Map selected coord back to original mesh index.
        try:
            selected_coord = self.coord_map[selected_coord]
        except KeyError:
            print("RAN OUT OF VERTICES")
            return 2

        # Need to convert selected coord into index of remaining coordinates
        selected_coord_idx = self.coordinate_list.index(selected_coord)

        # Get boundaries in mesh
        bmesh = BoundaryMesh(self.flow_solver.mesh, 'local')
        boundary_vertices = bmesh.entity_map(0).array()

        # Get coordinates and connections
        coords = self.flow_solver.mesh.coordinates()
        cells = self.flow_solver.mesh.cells()

        # Keep track of removed coordinates
        self.removed_coordinates.append(coords[selected_coord_idx])

        # Remove cells that use selected coordinate
        cells = cells[np.argwhere(cells != selected_coord_idx)[:,0]]
        cells[cells > selected_coord] -= 1
        boundary_vertices[boundary_vertices > selected_coord_idx] -= 1

        # Remove selected coordinate from coordinate array (probably can be a one-liner)
        coord_list = list(range(len(coords)))
        del coord_list[selected_coord_idx]
        del self.coordinate_list[selected_coord_idx]
        coords = coords[coord_list]

        # Use scipy Delaunay to remesh from vertices
        try:
            tri = Delaunay(coords)
        except ValueError: # It's bad if the mesh cannot be triangulated
            self.coordinate_list.insert(selected_coord, selected_coord)
            print("\nMESH BROKE, COULDN'T TRIANGULATE")
            return 2

        cells = tri.simplices

        # Remove all cells that only have boundary vertices
        cells = cells[np.sum(np.isin(cells, boundary_vertices), axis=1) != 3]

        # Create new mesh
        mesh = Mesh(self.flow_solver.mesh)
        editor = MeshEditor()
        editor.open(mesh, 'triangle', 2, 2)
        editor.init_vertices(len(coords))
        editor.init_cells(len(cells))

        for idx, vert in enumerate(coords):
            editor.add_vertex(idx, vert)
        for idx, c in enumerate(cells):
            editor.add_cell(idx, cells[idx].astype(np.uintp))
        editor.close()

        val = self._check_mesh(mesh, selected_coord)
        return val


    def _calculate_velocities(self):
        self.velocities = np.array([list(map(lambda x: u(x, allow_extrapolation=True),
                                    self.flow_solver.mesh.coordinates())) for u in self.u])


    def _calculate_pressures(self):
        self.pressures = np.array([list(map(lambda x: p(x, allow_extrapolation=True),
                                   self.flow_solver.mesh.coordinates())) for p in self.p])[:,:,np.newaxis]


    #@staticmethod
    def _interpolate(self, idx, original_u, original_p, v_func, p_func):
        # Interpolate to new values
        v_func.interpolate(original_u.copy(deepcopy=True))
        p_func.interpolate(original_p.copy(deepcopy=True))

        # Calculate new values
        u = v_func.copy(deepcopy=True)
        u.set_allow_extrapolation(True)
        self.u[idx] = u.copy(deepcopy=True)
        self.velocities[idx] = np.array(list(map(lambda x: u(x, allow_extrapolation=True), \
                                        self.flow_solver.mesh.coordinates())))

        p = p_func.copy(deepcopy=True)
        p.set_allow_extrapolation(True)
        self.p[idx] = p.copy(deepcopy=True)
        self.pressures[idx] = np.array(list(map(lambda x: p(x, allow_extrapolation=True), \
                                       self.flow_solver.mesh.coordinates())))[:,np.newaxis]
        del v_func
        del p_func


    def _check_mesh(self, mesh, selected_coord):
        # If the mesh didn't break
        if(selected_coord in self.removable):

            # Remesh
            old_mesh = Mesh(self.flow_solver.mesh)
            self.flow_solver.remesh(mesh)
            
            # Interpolate Velocities and pressures... somehow all the same?
            for idx, (original_u, original_p) in enumerate(zip(self.original_u, self.original_p)):
                try:
                    V_new = VectorFunctionSpace(self.flow_solver.mesh, 'Lagrange', 2)
                    #V_new = VectorFunctionSpace(self.flow_solver.mesh, 'Lagrange', 3)
                    v_func = Function(V_new, degree=2)
                    v_func.set_allow_extrapolation(True)
                    v_func.interpolate(original_u.copy(deepcopy=True))

                    P_new = FunctionSpace(self.flow_solver.mesh, 'Lagrange', 1)
                    #P_new = FunctionSpace(self.flow_solver.mesh, 'Lagrange', 3)
                    p_func = Function(P_new, degree=1)
                    p_func.set_allow_extrapolation(True)
                    p_func.interpolate(original_p.copy(deepcopy=True))
                except RuntimeError:
                    print("INTERPOLATION BROKE")
                    self.flow_solver.mesh = old_mesh
                    self.coordinate_list.insert(selected_coord, selected_coord)
                    return 2 # Node removal broke mesh

                try:
                    u = v_func.copy(deepcopy=True)
                    u.set_allow_extrapolation(True)
                    self.u[idx] = u.copy(deepcopy=True)

                    p = p_func.copy(deepcopy=True)
                    p.set_allow_extrapolation(True)
                    self.p[idx] = p.copy(deepcopy=True)
                except RuntimeError:
                    print("CALCULATION BROKE")
                    self.flow_solver.mesh = old_mesh
                    self.coordinate_list.insert(selected_coord, selected_coord)
                    return 2 # Node removal broke mesh

                del v_func
                del p_func

            self._calculate_velocities()
            self._calculate_pressures()

            # Update this to reflect removed vertex
            self.removable = np.argwhere(self.flow_solver.removable)[:,0]

            return 0
        else:
            self.coordinate_list.insert(selected_coord, selected_coord)
            print("\nMESH BROKE. SKIPPING VERTEX REMOVAL\n")
            return 2 # Node removal broke mesh


if __name__ == '__main__':
    with open("./configs/ag12.yaml", 'r') as stream:
        flow_config = yaml.safe_load(stream)

    env = ClosestEnv2DAirfoil(flow_config)

    import random
    state = env.get_state()
    for t in tqdm(range(10)):
    
        #action = select_action(state)
        action = torch.tensor([random.sample(range(env.N_CLOSEST), 1)], dtype=torch.long).to(device)
        next_state, reward, done, _ = env.step(action.item())
    
        reward = torch.tensor([reward])
    
        # Observe new state
        if(done):
            next_state = None
    
        state = next_state
