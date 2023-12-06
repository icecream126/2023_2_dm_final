import torch
from torch_geometric.nn import HeteroConv, SAGEConv, to_hetero
import torch.nn.functional as F
from torch_geometric.data import HeteroData

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
            ('paper', 'written_by', 'author'): SAGEConv((-1, -1), hidden_channels)
        })
        self.conv2 = HeteroConv({
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), out_channels),
            ('paper', 'written_by', 'author'): SAGEConv((-1, -1), out_channels)
        })

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

