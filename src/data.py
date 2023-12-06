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
from sklearn.model_selection import train_test_split


def process_data():

    papers_df = pd.read_csv('./dataset/processed_papers.csv')
    authors_df = pd.read_csv('./dataset/processed_authors.csv')

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
    # out_channels = 1
    
    '''Create train, valid, test mask'''
    # Create masks for train, validation, and test sets for authors
    train_idx, test_idx = train_test_split(range(len(authors_df)), test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Create train, val, test masks
    train_mask = torch.zeros(len(authors_df), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(authors_df), dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(len(authors_df), dtype=torch.bool)
    test_mask[test_idx] = True

    # Add masks to HeteroData
    data['author'].train_mask = train_mask
    data['author'].val_mask = val_mask
    data['author'].test_mask = test_mask

    
    
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

    data['author', 'writes', 'paper'].edge_index = edge_index
    data['paper', 'written_by', 'author'].edge_index = torch.flip(edge_index, [0])
    

    return data, out_channels, affiliation_encoder, test_mask