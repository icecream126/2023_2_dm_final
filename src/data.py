import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, HANConv
import torch.nn.functional as F
import torch.nn as nn
import torch

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from model import HANModel
import torch.nn as nn
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader


def process_data():

    papers_df = pd.read_csv('./dataset/gathered_paper.csv')
    authors_df = pd.read_csv('./dataset/gathered_author.csv')

    paper_encoder = LabelEncoder()
    author_encoder = LabelEncoder()
    papers_df['paper_id_encoded'] = paper_encoder.fit_transform(papers_df['paper_id']) # [12873, 14]
    authors_df['author_id_encoded'] = author_encoder.fit_transform(authors_df['author_id']) # [54819, 7]

    data = HeteroData()

    '''Create paper feature'''
    # paper encoding
    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    papers_df['combined_text'] = papers_df['paper_title'] + ' ' + papers_df['paper_area'] + ' ' + papers_df['paper_abstract']
    tfidf_embeddings = tfidf_vectorizer.fit_transform(papers_df['combined_text']).toarray()# numpy [12873, 100]
    data['paper'].x = torch.tensor(tfidf_embeddings, dtype=torch.float) # [12873, 100]

    '''Create author feature'''
    # author position one-hot encoding
    position_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse_output=False)
    positions_encoded = position_encoder.fit_transform(authors_df['author_position']) # [54819]
    positions_onehot = onehot_encoder.fit_transform(positions_encoded.reshape(-1, 1)) # [54819, 736]
    # author affiliation type
    affiliation_type = authors_df['author_affiliation_type'].values.reshape(-1, 1) # [54819, 1]

    author_features = np.hstack((positions_onehot, affiliation_type)) # [54819, 737]
    data['author'].x = torch.tensor(author_features, dtype=torch.float) # [54819, 737]

    '''Create author label'''
    # Add target labels for authors (affiliation)
    affiliation_encoder = LabelEncoder()
    authors_df['affiliation_encoded'] = affiliation_encoder.fit_transform(authors_df['author_affiliation']) # [54819]
    author_labels = torch.tensor(authors_df['affiliation_encoded'].values, dtype=torch.long) # [54819]
    data['author'].y = author_labels
    out_channels = len(affiliation_encoder.classes_)
    
    '''Create training mask'''
    num_authors = len(authors_df)
    train_mask = torch.zeros(num_authors, dtype=torch.bool)
    num_train_authors = int(len(authors_df)*0.8)
    train_mask[:num_train_authors] = True
    data['author'].train_mask = train_mask
    
    
    '''Create edge index'''
    paper_id_to_idx = {pid: idx for idx, pid in enumerate(papers_df['paper_id'])} # dict : {paper_id, index}
    author_id_to_idx = {aid: idx for idx, aid in enumerate(authors_df['author_id'])} # dict : {author_id, index}

    # Correct edge index construction
    edge_index = []
    for _, row in authors_df.iterrows(): # row : author dataframe에서 하나의 row (data)씩 읽어옴
        paper_idx = paper_id_to_idx.get(row['paper_id']) # get paper index via paper id
        author_idx = author_id_to_idx.get(row['author_id']) # get author index via author id

        if paper_idx is not None and author_idx is not None:
            edge_index.append([author_idx, paper_idx])  # Ensure indices are within valid range

    edge_index = torch.tensor(edge_index).t().contiguous() # [2, 54819]

    # edge_index = torch.tensor(edge_index).t().contiguous()
    data['author', 'writes', 'paper'].edge_index = edge_index
    data['paper', 'written_by', 'author'].edge_index = torch.flip(edge_index, [0])
    
    # train_loader, valid_loader, test_loader = get_loader(data = data, num_authors = len(authors_df), batch_size=128, num_neighbors=10)

    return data, out_channels, affiliation_encoder


# def get_loader(data, num_authors, batch_size, num_neighbors):
#     indices = torch.randperm(num_authors)

#     # Split indices for train, val, and test sets
#     train_end = int(num_authors * 0.8)
#     val_end = train_end + int(num_authors * 0.1)

#     train_indices = indices[:train_end]
#     val_indices = indices[train_end:val_end]
#     test_indices = indices[val_end:]

#     # Create masks
#     train_mask = torch.zeros(num_authors, dtype=torch.bool).scatter_(0, train_indices, True)
#     val_mask = torch.zeros(num_authors, dtype=torch.bool).scatter_(0, val_indices, True)
#     test_mask = torch.zeros(num_authors, dtype=torch.bool).scatter_(0, test_indices, True)

#     # Apply masks to data
#     data['author'].train_mask = train_mask
#     data['author'].val_mask = val_mask
#     data['author'].test_mask = test_mask
    
#     batch_size = batch_size
#     num_neighbors = num_neighbors  # Adjust the number of neighbors to sample
    
#     train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=('author', train_mask))
#     valid_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=('author', val_mask))
#     test_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=('author', test_mask))
    
#     return train_loader, valid_loader, test_loader

def get_loader(data, num_authors, batch_size, num_neighbors):
    indices = torch.randperm(num_authors)

    # Split indices for train, val, and test sets
    train_end = int(num_authors * 0.8)
    valid_end = train_end + int(num_authors * 0.1)

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    # Create masks
    train_mask = torch.zeros(num_authors, dtype=torch.bool).scatter_(0, train_indices, True)
    val_mask = torch.zeros(num_authors, dtype=torch.bool).scatter_(0, valid_indices, True)
    test_mask = torch.zeros(num_authors, dtype=torch.bool).scatter_(0, test_indices, True)

    # Apply masks to data
    data['author'].train_mask = train_mask
    data['author'].val_mask = val_mask
    data['author'].test_mask = test_mask
    
    # assert 'author' in data and 'paper' in data, "Data does not have required node types"
    assert hasattr(data['author'], 'x') and hasattr(data['paper'], 'x'), "Node features are missing"
    assert hasattr(data['author', 'writes', 'paper'], 'edge_index'), "Edge index for 'author-writes-paper' is missing"
    assert hasattr(data['paper', 'written_by', 'author'], 'edge_index'), "Edge index for 'paper-written_by-author' is missing"

    # Ensure indices are within valid range
    assert train_indices.max() < num_authors and valid_indices.max() < num_authors and test_indices.max() < num_authors, "Node indices are out of range"

    # # Function to create a subset of HeteroData
    # def create_subset(data, indices):
    #     subset = HeteroData()
    #     for key in data.keys():
    #         if 'author' in key:
    #             item = data[key]
    #             if isinstance(item, torch.Tensor):
    #                 subset[key] = item[indices]
    #             else:
    #                 subset[key] = item
    #     return subset

    # # Creating subsets of data for each loader
    # train_data = create_subset(data, train_indices)
    # valid_data = create_subset(data, valid_indices)
    # test_data = create_subset(data, test_indices)
        # Creating subsets of data for each loader
    train_data = create_subset(data, train_indices, 'author')
    valid_data = create_subset(data, valid_indices, 'author')
    test_data = create_subset(data, test_indices, 'author')

    
    # Creating DataLoader for each dataset
    train_loader = NeighborLoader(train_data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=('author', np.array(data['author'].train_mask)))
    valid_loader = NeighborLoader(valid_data, num_neighbors=num_neighbors, batch_size=batch_size, size=len(valid_indices))
    test_loader = NeighborLoader(test_data, num_neighbors=num_neighbors, batch_size=batch_size, size=len(test_indices))
    
    return train_loader, valid_loader, test_loader

def create_subset(data, indices, node_type):
    # Create a new HeteroData object
    subset = HeteroData()

    # Copy node features for the specified node type
    for key, value in data[node_type].items():
        if key == 'x' or key == 'y':
            subset[node_type][key] = value[indices]
        else:
            subset[node_type][key] = value

    # Copy edge features for edges connected to the specified node type
    for edge_type in data.edge_types:
        if node_type in edge_type:
            for key, value in data[edge_type].items():
                # Assuming edge indices don't need to be subsetted
                subset[edge_type][key] = value

    return subset

# def create_subset(data, indices, node_type):
#     subset = HeteroData()

#     # Copy node features for the specified node type
#     subset[node_type].x = data[node_type].x[indices]
#     subset[node_type].y = data[node_type].y[indices]

#     # Copy edge indices and other features, ensuring they refer to valid nodes
#     for edge_type in data.edge_types:
#         if node_type in edge_type:
#             edge_index = data[edge_type].edge_index
#             mask = torch.isin(edge_index[0 if node_type == edge_type[0] else 1], indices)
#             for key, value in data[edge_type].items():
#                 if key == 'edge_index':
#                     # Adjust edge indices to new node indexing
#                     new_edge_index = edge_index[:, mask]
#                     if node_type == edge_type[0]:  # if node_type is src
#                         new_edge_index[0] = torch.argsort(torch.argsort(indices))[new_edge_index[0]]
#                     else:  # if node_type is dst
#                         new_edge_index[1] = torch.argsort(torch.argsort(indices))[new_edge_index[1]]
#                     subset[edge_type][key] = new_edge_index
#                 else:
#                     subset[edge_type][key] = value[mask]

#     # Ensure the number of nodes is correctly set for each node type
#     for node_type in data.node_types:
#         subset[node_type].num_nodes = len(indices) if node_type == 'author' else data[node_type].num_nodes

#     return subset

