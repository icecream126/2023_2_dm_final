from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HANConv


class GAT(nn.Module):
    def __init__(self, dim_in, dim_h, out_channels, heads, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(dim_in, dim_h, heads, dropout=dropout)
        self.conv2 = GATConv(dim_h * heads, out_channels, heads=1,
                             concat=False, dropout=dropout)

    def forward(self, x, edge_index, dropout=0.6):
        x = F.dropout(x, p=dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class HAN(nn.Module):
    def __init__(self, data, dim_in, dim_out, dropout=0.6, dim_h=512, heads=8):
        super().__init__()
        self.han = HANConv(dim_in, dim_h, heads=heads, dropout=dropout, metadata=data.metadata())
        self.linear = nn.Linear(dim_h, dim_out)
 
    def forward(self, x_dict, edge_index_dict):
        out = self.han(x_dict, edge_index_dict)
        out = self.linear(out['author'])
        return out