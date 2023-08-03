from typing import Optional
import torch
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import DataLoader
import torch_geometric
import torch_scatter
from utils import *

class MeshGraphNet(torch.nn.Module):
    #x: node features
    #edge_index: connectivity of edges
    #edge_attr: edge features(relative velocity)
    #y: node output
    #p: pressure
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, args, emb=False):
        super(MeshGraphNet, self).__init__()

        #encoder
        self.node_encoder = nn.Sequential(nn.Linear(input_dim_node, hidden_dim), 
                                           nn.ReLU(), 
                                           nn.Linear(hidden_dim, hidden_dim),
                                           nn.LayerNorm(hidden_dim))
        
        self.edge_encoder = nn.Sequential(nn.Linear(input_dim_edge, hidden_dim), 
                                           nn.ReLU(), 
                                           nn.Linear(hidden_dim, hidden_dim),
                                           nn.LayerNorm(hidden_dim))
        
        self.num_layers = args.num_layers

        #processor
        self.processor = torch.nn.ModuleList()
        processor_layer = ProcessorLayer(in_channel=hidden_dim, out_channel=hidden_dim)
        for i in range(args.num_layers):
            self.processor.append(processor_layer)

        #decoder
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, output_dim))
        
    
    def forward(self, data, mean_vec, std_vec, mean_edge_vec, std_edge_vec):
        x, edge_index, edge_attr, p = data.x, data.edge_index, data.edge_attr, data.p

        #normalize
        x = normalize(x, mean_vec, std_vec)
        edge_attr = normalize(edge_attr, mean_edge_vec, std_edge_vec)

        #encode
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        #process
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        #decode
        x = self.decoder(x)
        return x

    def loss(self, pred, input, mean_vec, std_vec):
        labels = normalize(input.y, mean_vec, std_vec)
        loss_mask=torch.logical_or((torch.argmax(input.x[:,2:],dim=1)==torch.tensor(0)),
                                   (torch.argmax(input.x[:,2:],dim=1)==torch.tensor(5)))
        error = torch.sum((pred - labels)**2, dim=1)
        loss = torch.sqrt(torch.mean(error[loss_mask]))
        return loss



class ProcessorLayer(MessagePassing):
    def __init__(self, in_channel=128, out_channel=128, **kwargs):
        super(ProcessorLayer, self).__init__(  **kwargs)
        self.node_mlp = nn.Sequential(nn.Linear(2*in_channel, out_channel), 
                                       nn.ReLU(), 
                                       nn.Linear(out_channel, out_channel),
                                       nn.LayerNorm(out_channel))
        self.edge_mlp = nn.Sequential(nn.Linear(3*in_channel, out_channel),
                                       nn.ReLU(),
                                       nn.Linear(out_channel, out_channel),
                                       nn.LayerNorm(out_channel))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.node_mlp[0].reset_parameters()
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()
        self.edge_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        out, updated_edges =  self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = x + self.node_mlp(updated_nodes)
        return updated_nodes, updated_edges
    
    def message(self, x_i, x_j, edge_attr):
        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)
        updated_edges = self.edge_mlp(updated_edges) + edge_attr
        return updated_edges
    
    def aggregate(self, updated_edges, edge_index, dim_size=None):
        out = torch_scatter.scatter(updated_edges, edge_index[0,:], dim=0, dim_size=dim_size)
        return out, updated_edges