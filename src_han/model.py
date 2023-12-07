from torch_geometric.nn import HANConv
import torch.nn.functional as F
import torch.nn as nn

    
class HANModel(nn.Module):
    def __init__(self, data, dim_in, dim_out, dropout=0.6, dim_h=512, heads=8):
        super().__init__()
        self.han = HANConv(dim_in, dim_h, heads=heads, dropout=dropout, metadata=data.metadata())
        self.linear = nn.Linear(dim_h, dim_out)
 
    def forward(self, x_dict, edge_index_dict):
        out = self.han(x_dict, edge_index_dict)
        out = self.linear(out['author'])
        return out