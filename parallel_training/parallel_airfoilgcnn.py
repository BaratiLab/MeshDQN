import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GraphConv, TopKPooling,  GCNConv, avg_pool, TAGConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import tqdm
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import KFold
import ray

# Set torch device
if(torch.cuda.is_available()):
    print("USING GPU")
    device = torch.device("cuda:0")
else:
    print("USING CPU")
    device = torch.device("cpu")
device = torch.device("cpu")

#@ray.remote
class NodeRemovalNet(torch.nn.Module):
    def __init__(self, output_dim, conv_width=64, topk=0.5, initial_num_nodes=None):

        super(NodeRemovalNet, self).__init__()
        self.conv_width = conv_width
        self.initial_num_nodes = initial_num_nodes
        self.conv1 =  SAGEConv(2, conv_width).float()
        self.pool1 = TopKPooling(conv_width, ratio=topk).float()
        self.conv2 =  SAGEConv(conv_width, conv_width).float()
        self.pool2 = TopKPooling(conv_width, ratio= topk).float()
        self.conv3 =  SAGEConv(conv_width, conv_width).float()
        self.pool3 = TopKPooling(conv_width, ratio=topk).float()
        self.conv4 =  GCNConv(conv_width, conv_width).float()
        self.pool4 = TopKPooling(conv_width, ratio=topk).float()
        self.conv5 =  GCNConv(conv_width, conv_width).float()
        self.pool5 = TopKPooling(conv_width, ratio=topk).float()
        self.conv6 =  GCNConv(conv_width, conv_width).float()
        self.pool6 = TopKPooling(conv_width, ratio=topk).float()
        self.lin1 = torch.nn.Linear(2*conv_width, 128).float()
        self.lin2 = torch.nn.Linear(128, 64).float()
        self.lin3 = torch.nn.Linear(64, output_dim).float()

        torch.manual_seed(0)
        self.reset()
        

    def reset(self):
        # SAGEConv layers
        nn.init.xavier_normal_(self.conv1.lin_l.weight, gain=0.9)
        nn.init.normal_(self.conv1.lin_l.bias)
        nn.init.xavier_normal_(self.conv1.lin_r.weight, gain=0.9)

        nn.init.xavier_normal_(self.conv2.lin_l.weight, gain=0.9)
        nn.init.normal_(self.conv2.lin_l.bias)
        nn.init.xavier_normal_(self.conv2.lin_r.weight, gain=0.9)

        nn.init.xavier_normal_(self.conv3.lin_l.weight, gain=0.9)
        nn.init.normal_(self.conv3.lin_l.bias)
        nn.init.xavier_normal_(self.conv3.lin_r.weight, gain=0.9)

        # GCN layers
        nn.init.xavier_normal_(self.conv4.lin.weight, gain=0.9)
        nn.init.xavier_normal_(self.conv5.lin.weight, gain=0.9)
        nn.init.xavier_normal_(self.conv6.lin.weight, gain=0.9)

        nn.init.xavier_normal_(self.lin1.weight, gain=0.9)
        nn.init.normal_(self.lin1.bias)

        nn.init.xavier_normal_(self.lin2.weight, gain=0.9)
        nn.init.normal_(self.lin2.bias)

        nn.init.xavier_normal_(self.lin3.weight, gain=0.9)
        nn.init.normal_(self.lin3.bias)
        
    def set_num_nodes(self, initial_num_nodes):
        self.initial_num_nodes = initial_num_nodes
        self.conv1 =  SAGEConv(self.initial_num_nodes, self.conv_width).to(device)

    def set_removable(self, removable):
        self.removable = removable

    def forward(self, data, embedding=False):
        """
        data: Batch of Pytorch Geometric data objects, containing node features, edge indices and batch size
            
        returns: Predicted normalized drag value
        """
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        #print(x.shape)

        x = F.relu(self.conv1(x, edge_index))
        #print(x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        #print(x.shape)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        #print(x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        #print(x.shape)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        #x = F.relu(self.conv3(x, edge_index))
        #print(x.shape)
        #x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        #print(x.shape)
        #x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index))
        #print(x.shape)
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        #print(x.shape)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv5(x, edge_index))
        #print(x.shape)
        x, edge_index, _, batch, _, _ = self.pool5(x, edge_index, None, batch)
        #print(x.shape)
        x5 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #x = F.relu(self.conv6(x, edge_index))
        #print(x.shape)
        #x, edge_index, _, batch, _, _ = self.pool6(x, edge_index, None, batch)
        #print(x.shape)
        #x6 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape)
        #raise

        #x = x1+x2+x3+x4+x5+x6
        x = x1+x2+x4+x5

        if(embedding):
            return x

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.0, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = F.softmax(x, dim=1) # Pick which vertex to remove

        return x

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class AirfoilGCNN(torch.nn.Module):
    def __init__(self, conv_width=64):

        super(AirfoilGCNN, self).__init__()
        #conv_width = 64
        topk = 0.5
        self.conv1 =  SAGEConv(2, conv_width)
        self.pool1 = TopKPooling(conv_width, ratio=topk)
        self.conv2 =  SAGEConv(conv_width, conv_width)
        self.pool2 = TopKPooling(conv_width, ratio= topk)
        self.conv3 =  SAGEConv(conv_width, conv_width)
        self.pool3 = TopKPooling(conv_width, ratio=topk)
        self.conv4 =  GCNConv(conv_width, conv_width)
        self.pool4 = TopKPooling(conv_width, ratio=topk)
        self.conv5 =  GCNConv(conv_width, conv_width)
        self.pool5 = TopKPooling(conv_width, ratio=topk)
        self.conv6 =  GCNConv(conv_width, conv_width)
        self.pool6 = TopKPooling(conv_width, ratio=topk)
        self.lin1 = torch.nn.Linear(2*conv_width, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)

    def forward(self, data):
        """
        data: Batch of Pytorch Geometric data objects, containing node features, edge indices and     batch size

        returns: Predicted normalized drag value
        """
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = x[:,[2,3]]

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #print(x.shape)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv5(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool5(x, edge_index, None, batch)
        x5 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv6(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool6(x, edge_index, None, batch)
        x6 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1+x2+x3+x4+x5+x6

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.0, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


#@ray.remote
#class ParameterServer(object):
#    def __init__(self, keys, values):
#        # These values will be mutated, so we must create a local copy.
#        values = [value.copy() for value in values]
#        self.parameters = dict(zip(keys, values))
#    
#    def get(self, keys):
#        return self.parameters[key] for key in keys]
#
#    def update(self, keys, vlaues):
#        for key, value in zip(keys, values):
#            self.parameters[key] += value

