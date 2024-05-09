"""
This is an example of a 1D square problem. You can run this .py file to train a model for the square problem.
"""

# The corresponding installation package
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import (
    Any, Dict, List, Optional, Mapping, Set, TypeVar, Tuple, Union, Iterator, Sequence
)

from torch_geometric.nn import GCN2Conv, GATConv, GATv2Conv, GINConv, SAGEConv, SGConv, ARMAConv, GatedGraphConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

# random seed
import os
torch.manual_seed(123)
np.random.seed(123)
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

#load data , cfl=0.6
data_train = np.load(r'linear_1d.npy')
# time_steps*shapes
#coarsen to cfl=10.2
data_train_coarsen = []
for i in range(data_train.shape[0]//17+1):
    data_train_coarsen.append(data_train[17*i])
data_train_coarsen = np.stack(data_train_coarsen,axis=0)

# generate the training data

def make_train_data(data, output_time_step=1, delta_t=0.025, times=2):
    time_steps_all = data.shape[0] - 1
    train_input = {}
    train_input_sol = data[:-output_time_step * times]
    print(train_input_sol.shape)
    current_time = np.array([[delta_t * j] * data.shape[1] for j in range(train_input_sol.shape[0])])
    current_time = current_time.reshape(-1, 1)
    train_input['current_time'] = current_time
    train_input['time_step'] = np.ones(current_time.shape[0]) * delta_t * times

    train_input_sol = train_input_sol.reshape(-1, data.shape[-1])

    # train_input_sol = train_input_sol[..., np.newaxis]
    train_input['solution'] = train_input_sol
    train_output = []
    for shift in range(times, output_time_step * times + 1, times):
        output_slice = data[shift:time_steps_all - output_time_step * times + shift + 1]
        output_slice = output_slice.reshape(-1, data.shape[-1])
        output_slice = output_slice[..., np.newaxis]
        train_output.append(output_slice)
    train_output = np.concatenate(train_output, axis=2)
    assert train_input['solution'].shape[0] == train_output.shape[0]
    assert train_input['solution'].shape[1] == train_output.shape[1]
    return train_input, train_output
    
# you can also use this func to generate the training data consisting of several CFL numbers
def make_train_data_multi(data, output_time_step=2, delta_t=1/32*0.6*17, times_steps_all=[3,4,5]):
    train_input = {}
    train_output = []
    current_time = []
    time_step = []
    train_input_sol = []
    for k in times_steps_all:
        train_input_temp, train_output_temp = make_train_data(data, output_time_step, delta_t, k)
        current_time.append(train_input_temp['current_time'])
        time_step.append(train_input_temp['time_step'])
        train_input_sol.append(train_input_temp['solution'])
        train_output.append(train_output_temp)
    train_input['solution'] = np.concatenate(train_input_sol, axis=0)
    train_input['current_time'] = np.concatenate(current_time, axis=0)
    train_input['time_step'] = np.concatenate(time_step, axis=0)
    train_output = np.concatenate(train_output, axis=0)

    return train_input, train_output



train_input, train_output = make_train_data_multi(data_train_coarsen[:20],1,times_steps_all = [1])#cfl=10.2


# build the model
class Grid():
    def __init__(self, size:int=32, domain:Tuple[float,float] = (0.,1.)):
        self.size = size
        self.domain = domain
        self.length = domain[1] - domain[0]
        assert self.length > 0
        self.step:float = self.length/self.size

        self.x = np.linspace(self.domain[0],self.domain[1], self.size + 1)

        self.xc = np.arange(self.domain[0]+self.step*0.5, self.domain[1], self.step)

def velocity(x,t):
    if not isinstance(t, np.ndarray):
        t = np.array(t) 


    return np.ones_like(x,dtype=np.float64)

class SLGeometry:
    def __init__(self, grid: Grid, vel=velocity):
        self.grid = grid
        self.velocity = vel

    def get_upstream(self, t, time_step):  # t: current time
        
        Nt = t.shape[0]
        X = np.stack([self.grid.xc] * Nt, axis=0)
        T = t + time_step

        k1 = time_step[:, np.newaxis] * self.velocity(X, T)
        k2 = time_step[:, np.newaxis] * self.velocity(X - 0.5 * k1, T - 0.5 * time_step)
        k3 = time_step[:, np.newaxis] * self.velocity(X - 0.5 * k2, T - 0.5 * time_step)
        k4 = time_step[:, np.newaxis] * self.velocity(X - k3, T - time_step)

        return - (k1 + 2 * k2 + 2 * k3 + k4) / 6.

# constrcut the edges
def generate_index_1d(x_grid, x_new):
    """
    Find the position of the upstream point in 1D
    """
    batch_size = x_new.shape[0]
    num_points = x_new.shape[1]
    indices = np.zeros((batch_size, num_points), dtype=int)
    x_gird_repeated = np.stack([x_grid] * num_points, axis=0)
    x_gird_repeated = np.stack([x_gird_repeated] * batch_size, axis=0)
    x_new_repeated = np.tile(x_new[...,np.newaxis], (1, 1, num_points))
    
    indices = np.argmin(x_gird_repeated <= x_new_repeated, axis=2)-1# 
   
    return indices

grid = Grid()
x_grid = grid.xc # the grid points

# the indexs for each grid point in the graph
node_index = np.arange(0,32).reshape(32)

def generate_edge(indices_all, grid_index = node_index):
    """
    Find the index of the edge based on the position of the upstream point
    in this case we use the nearest 2 points to construct edges
    """
    batch_size = indices_all.shape[0]
    num_x = indices_all.shape[1]
    
    
    edge_p = np.zeros((2, 2, batch_size, num_x), dtype=int)
    edge_p[1] = np.tile(grid_index,(2,batch_size,1))
    
    edge_p[0,0] = (indices_all[:,:])%32
    edge_p[0,1] = (indices_all[:,:]+1)%32

    return edge_p.transpose(2,0,3,1)

def generate_edge_from_xi(xi_n):
    """
    from xi(non norm) to the corresponding edges 
    """
    x_n = x_grid + xi_n
    x_n = x_n - (x_n // 1) * (1) 
    indx = generate_index_1d(x_grid, x_n)
    
    indx = indx%32
    edges_al = generate_edge(indx)
    edges_al = edges_al.reshape(edges_al.shape[0],2,-1)# batch*2*len(edges)
    
    return edges_al

    
class Conv1DPeriodic(nn.Module):
    def __init__(self, filters1, filters2, kernel_size, activation):
        super(Conv1DPeriodic, self).__init__()
        self._layer = nn.Conv1d(filters1, filters2, kernel_size, padding='valid')
        self.activation = getattr(F,activation)
        self.kernel_size = kernel_size

        # if any(size % 2 == 0 for size in kernel_size):
        #     raise ValueError('kernel size for conv1d is not odd: {}'.format(kernel_size))

    def forward(self, inputs):
        padding = self.kernel_size // 2
        padded = self.pad_periodic_1d(inputs, padding)
        # print(padded.shape)
        result = self._layer(padded)
        result = self.activation(result)
        return result

    def pad_periodic_1d(self, inputs, padding):
        tensors = [inputs[:, :,-padding:], inputs, inputs[:, :,:padding]]
        return torch.cat(tensors, dim=2)

def conv1d_stack(num_inputs,num_outputs, num_layers=6, filters=32, kernel_size=5,
                 activation='elu', **kwargs):
    """Create a sequence of Conv2DPeriodic layers."""
    model = nn.Sequential()
    model.add_module('conv_layer_first', Conv1DPeriodic(num_inputs,filters, kernel_size, activation, **kwargs))
    for _ in range(num_layers-2):
        layer = Conv1DPeriodic(
            filters, filters,kernel_size, activation, **kwargs)
        model.add_module(f'conv_layer{_}', layer)
    model.add_module('conv_layer_last', Conv1DPeriodic(filters,num_outputs, kernel_size, activation, **kwargs))
    # model.add_module('output_layer', OutPutLayer(num_outputs)
    return model

# construct the graph list 
data_list = []
# Convert input and output into graph data
sol = []
for i in range(train_input['solution'].shape[0]):
    sol.append(train_input['solution'][i].reshape(-1,1))
train_output = train_output.reshape(train_output.shape[0],train_output.shape[1],-1)
for i in range(train_output.shape[0]):
    # data 
    data = Data(x=torch.tensor(sol[i]),y=torch.tensor(train_output[i]))
    data.cur_time = torch.tensor(train_input['current_time'][[i]])
    data.time_step = torch.tensor(train_input['time_step'][[i]])
    data_list.append(data)

#  batch
batch_size = 4
loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    
# create a global model consisting of a sequence of convolutional layers and graph attention layers
class GlobalModel(torch.nn.Module):
    def __init__(self, num_steps, num_inputs,num_features, num_outputs,conv_layers,hidden_features,heads):
        super(GlobalModel, self).__init__()
        self.num_steps = num_steps
        self.slmodel = SLGeometry(Grid())
        self.conv = conv1d_stack(num_inputs, num_features, num_layers=conv_layers)# 
        self.graph_layer1 = GATv2Conv(num_features, hidden_features, heads=heads)
        self.graph_layer2 = GATv2Conv(hidden_features*heads, hidden_features, heads=heads)
        self.linear1 = nn.Linear(2*hidden_features*heads,num_outputs)
        
    def forward(self,data):
        x = data.x.to(device)
        cur_time, time_step = data.cur_time.numpy(), data.time_step.numpy()
        pred_all = []
        for k in range(self.num_steps):
            
            xi = self.slmodel.get_upstream(cur_time, time_step)

            edge_index = generate_edge_from_xi(xi)
            edge_index = torch.tensor(edge_index)
            data_new = Batch.from_data_list([Data(x=x[i*32:(i+1)*32],edge_index=edge_index[i]) for i in range(data.num_graphs)])
            edge_index = data_new.edge_index.to(device)
           
            xi = torch.tensor(xi/self.slmodel.grid.step).to(device)
            
            xc = x.reshape(data.num_graphs,32)

            xc = self.conv(torch.stack([torch.squeeze(xc), xi], dim=1))
            x_t = torch.transpose(xc,1,2)
            xc = x_t.reshape(data.num_graphs*32,-1)
            
            xc = self.graph_layer1(xc, edge_index)
            xc = self.graph_layer2(xc, edge_index)
            xc = F.elu(xc)
            row,col = edge_index
            
            edge_features = torch.cat([xc[row],xc[col]],dim=1)
            edge_features = self.linear1(edge_features)
            #constraint layer
            row_mask = row.unsqueeze(1) == torch.arange(x.shape[0]).unsqueeze(0).to(device)
            edge_mask = edge_features*row_mask
            edge_sum = torch.sum(edge_mask,dim=0)
            mask = torch.sum(row_mask,dim = 0)
            edge_features = edge_mask-row_mask*edge_sum/mask+1./mask*row_mask
            edge_features = torch.sum(edge_features,dim=1,keepdim=True)
            
            mask = col.unsqueeze(1) == torch.arange(x.shape[0]).unsqueeze(0).to(device)
            x_new = torch.sum(x[row]*mask*edge_features,dim=0,keepdim=True)
            x = x_new.T
            pred_all.append(x)
            del mask, edge_features, edge_sum, edge_mask, row_mask
            cur_time += time_step[:,np.newaxis]
        return torch.concat(pred_all, dim=1)



model = GlobalModel(num_steps=1, num_inputs=2, num_features=32, num_outputs=1,conv_layers=6, hidden_features=32,heads=4).to(torch.float64)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=0.001, weight_decay=5e-4)

scheduler = ExponentialLR(optimizer, gamma=0.98)
# if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
def train():
    model.train()
    loss_all = 0
    start_time = time.time()
    for data in loader:
        optimizer.zero_grad()
        output = model(data)
        
        
        loss = criterion(output, data.y.to(device))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    end_time = time.time()  # 记录结束时间
    epoch_time = end_time - start_time  
    print('Epoch {:03d}, Loss: {:.8f}, Time: {:.2f} seconds'.format(epoch, loss, epoch_time)) 
    return loss_all / len(loader.dataset)

for epoch in range(1, 101):
    loss = train()
    scheduler.step()  # update the lr

# after trained
torch.save(model.state_dict(), './weights/model_weights_square1d.pth') # cfl=10.2 