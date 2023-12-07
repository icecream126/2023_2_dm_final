from torch_geometric.nn import HANConv
import torch.nn.functional as F
import torch.nn as nn

# class HANLayer(nn.Module):
#     def __init__(self, paper_in_channels, author_in_channels, paper_out_channels, author_out_channels):
#         super(HANLayer, self).__init__()
#         self.conv_author_to_paper = GATConv(author_in_channels, paper_out_channels, add_self_loops=False)
#         self.conv_paper_to_author = GATConv(paper_out_channels, author_out_channels, add_self_loops=False)

#     def forward(self, x_dict, edge_index_dict):
#         paper_feats = self.conv_author_to_paper(x_dict['author'], edge_index_dict[('author', 'to', 'paper')])
#         author_feats = self.conv_paper_to_author(paper_feats, edge_index_dict[('paper', 'to', 'author')])

#         return {'author': author_feats, 'paper': paper_feats}

# class HANModel(nn.Module):
#     def __init__(self, paper_in_channels, author_in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.layer1 = HANLayer(paper_in_channels, author_in_channels, hidden_channels, hidden_channels)
#         self.layer2 = HANLayer(hidden_channels, hidden_channels, hidden_channels, out_channels)

#     def forward(self, x_dict, edge_index_dict):
#         x_dict = self.layer1(x_dict, edge_index_dict)
#         x_dict = {key: F.relu(x) for key, x in x_dict.items()}
#         x_dict = self.layer2(x_dict, edge_index_dict)
#         return x_dict
    
    
class HANModel(nn.Module):
    def __init__(self, data, dim_in, dim_out, dropout=0.6, dim_h=512, heads=8):
        super().__init__()
        self.han = HANConv(dim_in, dim_h, heads=heads, dropout=dropout, metadata=data.metadata())
        self.linear = nn.Linear(dim_h, dim_out)
 
    def forward(self, x_dict, edge_index_dict):
        out = self.han(x_dict, edge_index_dict)
        out = self.linear(out['author'])
        return out