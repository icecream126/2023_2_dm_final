from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn


    
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=dropout)

    def forward(self, x, edge_index, dropout=0.6):
        x = F.dropout(x, p=dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
