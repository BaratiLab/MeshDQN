from gym import Env, spaces
from flow_solver import FlowSolver
from probes import DragProbe, LiftProbe, PenetratedDragProbe
import numpy as np
import yaml

from dolfin import *
from matplotlib import pyplot as plt

from airfoilgcnn import AirfoilGCNN

from scipy.spatial import Delaunay

from tqdm import tqdm
import time

import torch
from torch import nn

from torch_geometric.data import Data
import os
from shapely.geometry import Polygon, Point

if(torch.cuda.is_available()):
    print("USING GPU")
    device = torch.device("cuda:0")
else:
    print("USING CPU")
    device = torch.device("cpu")
#device = torch.device("cpu")

class Env2DAirfoil(Env):
    """
        Environment to optimize mesh around a 2D airfoil
    """

    def __init__(self, config):
        super(Env2DAirfoil, self).__init__()

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

        # Solver Parameters
        self.gt_drag = config['agent_params']['gt_drag']
        self.gt_time = config['agent_params']['gt_time']
        self.u = config['agent_params']['u']
        self.p = config['agent_params']['p']
        self.original_u = config['agent_params']['u']
        self.original_p = config['agent_params']['p']

        # For putting vertices back after smoothing
        self.b_coords = self.flow_solver.bmesh.coordinates()

        self.out_of_vertices = False

        self.reset()


    def reset(self):
        drags, lifts = [], []
        if((self.gt_drag == -1) and (self.gt_time == -1)):
            print("CALCULATING INITIAL VALUE...")
            start = time.time()
            for i in tqdm(range(self.solver_steps)):
                u, p, drag, lift = self.flow_solver.evolve()
                drags.append(drag)
                lifts.append(lift)
            self.gt_drag = np.mean(drags[-1])
            self.gt_lift = lifts[-1]
            self.gt_time = time.time() - start
            self.u = u
            self.p = p
            self.original_u = u
            self.original_p = p

        # Get and save velocities
        self._calculate_velocities()
        self._calculate_pressures()

        # For keeping track of progress
        self.steps = 0
        self.num_episodes = 0
        self.terminal = False
        self._get_distance_lookup()


    def _snap_boundaries(self):
        print(self.flow_solver.mesh.coordinates())


    def _get_distance_lookup(self):
        removable = np.array(self.flow_solver.removable, dtype=int)
        coords = self.flow_solver.mesh.coordinates()
        not_removable = np.argwhere(-(np.array(self.flow_solver.removable)-1))[:,0]
        boundary_coords = coords[not_removable]
        airfoil_coords = boundary_coords[
                np.logical_and((boundary_coords[:,0] > -0.5) ,
                np.logical_and((boundary_coords[:,0] < 3),
                np.logical_and((boundary_coords[:,1] > -0.5),
                (boundary_coords[:,1] < 0.5))))
        ]
        self.polygon = Polygon(airfoil_coords)

        # Precompute distances for later lookup -> This will not work with mesh smoothing as its currently implemented
        self.distance_lookup = {}
        for idx, coord in enumerate(coords[self.removable]):
            self.distance_lookup[idx] = self.polygon.distance(Point(coord))


    def get_state(self):
        # Turn current mesh into PyG data object for AirfoilGCNN
        edge_index, edge_attr = [], []

        # calculate N closest point to the airofil
        self._n_closest()
        #if(self.out_of_vertices):
        #    return False

        # Only retain edge if its the N-closest 
        nstart = time.time()
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
        x = torch.Tensor(self.flow_solver.mesh.coordinates()[self.n_closest])
        x = torch.hstack((x, torch.Tensor(self.velocities[self.n_closest])))
        x = torch.hstack((x, torch.Tensor(self.pressures[self.n_closest])))

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)


    def _n_closest(self):
        # Keep track of mesh vertices
        self.coordinate_list = list(range(len(self.flow_solver.mesh.coordinates())))
        self.removable = np.argwhere(self.flow_solver.removable)[:,0] # Can now check if mesh broke from this

        # Map from removable vertex index to original index
        vec_lookup = np.vectorize(lambda x: self.coordinate_list.index(x))
        self.mesh_map = dict(zip(range(len(self.removable)), vec_lookup(self.removable)))
        coords = self.flow_solver.mesh.coordinates()

        distances = []
        self._get_distance_lookup()
        for idx, coord in enumerate(coords[self.removable]):
            distances.append(self.distance_lookup[idx])
        dist_idxs = np.argsort(distances)
        
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
            new_drag = self.flow_solver.drag_probe.sample(self.u, self.p)
            new_lift = self.flow_solver.lift_probe.sample(self.u, self.p)
            self.new_drag = new_drag
            self.new_lift = new_lift
        except:
            print("\n\nSAMPLING BROKE\n\n")
            return self.NEGATIVE_REWARD, True, True

        # Drag reward
        drag_factor = np.floor(-2*np.log(0.5)/self.threshold)
        drag_reward = 2*np.exp(-drag_factor*np.abs(self.gt_drag - new_drag)/np.abs(self.gt_drag)) - 1

        # Time reward
        time_reward = (self.initial_num_node - len(self.coordinate_list)) * self.TIME_REWARD

        if(drag_reward == np.nan):
            print(self.get_state())
            print("\n\nDRAG IS NAN!\n\n")
            raise
        return drag_reward+time_reward, False, \
               np.abs(np.abs(self.gt_drag - new_drag)/self.gt_drag) > self.threshold


    def set_plot_dir(self, plot_dir):
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)


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

        #adding_time = time.time()
        for idx, vert in enumerate(coords):
            editor.add_vertex(idx, vert)
        for idx, c in enumerate(cells):
            editor.add_cell(idx, cells[idx].astype(np.uintp))
        editor.close()

        val = self._check_mesh(mesh, selected_coord)
        return val


    def _calculate_velocities(self):
        self.velocities = np.array(list(map(lambda x: self.u(x, allow_extrapolation=True),
                                           self.flow_solver.mesh.coordinates())))


    def _calculate_pressures(self):
        self.pressures = np.array(list(map(lambda x: self.p(x, allow_extrapolation=True),
                                           self.flow_solver.mesh.coordinates())))[:,np.newaxis]


    def _check_mesh(self, mesh, selected_coord):
        # If the mesh didn't break
        if(selected_coord in self.removable):

            # Remesh
            old_mesh = Mesh(self.flow_solver.mesh)
            self.flow_solver.remesh(mesh)

            # Interpolate Velocities and pressures
            try:
                V_new = VectorFunctionSpace(self.flow_solver.mesh, 'Lagrange', 3)
                v_func = Function(V_new, degree=2)
                v_func.set_allow_extrapolation(True)
                v_func.interpolate(self.original_u.copy(deepcopy=True))

                P_new = FunctionSpace(self.flow_solver.mesh, 'Lagrange', 3)
                p_func = Function(P_new, degree=1)
                p_func.set_allow_extrapolation(True)
                p_func.interpolate(self.original_p.copy(deepcopy=True))
            except RuntimeError:
                print("INTERPOLATION BROKE")
                self.flow_solver.mesh = old_mesh
                self.coordinate_list.insert(selected_coord, selected_coord)
                return 2 # Node removal broke mesh

            try:
                self.u = v_func.copy(deepcopy=True)
                self._calculate_velocities()
                self.u.set_allow_extrapolation(True)

                self.p = p_func.copy(deepcopy=True)
                self.p.set_allow_extrapolation(True)
                self._calculate_pressures()
            except RuntimeError:
                print("CALCULATION BROKE")
                self.flow_solver.mesh = old_mesh
                self.coordinate_list.insert(selected_coord, selected_coord)
                return 2 # Node removal broke mesh

            del v_func
            del p_func

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
