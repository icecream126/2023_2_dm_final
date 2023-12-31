import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import VirtualNode

import os

def process_data(label_num=10, seed=0, feature_dim=300, sample_type='undersample'):
    
    savepath = f'./dataset/graph/{feature_dim}/{sample_type}/{label_num}_label/data.pt'
    df = pd.read_csv(f'./dataset/merged/data_{feature_dim}.csv')
    
    # Create paper feature
    filtered_columns = [col for col in df.columns if col.startswith('paper_abstract_filtered_')]
    filtered_data = df[filtered_columns]
    tensor_data = torch.tensor(filtered_data.values, dtype=torch.float32)

    # Create author label
    affiliation_encoder = LabelEncoder()
    df['affiliation_encoded'] = affiliation_encoder.fit_transform(df['author_affiliation']) 
    author_labels = torch.tensor(df['affiliation_encoded'].values, dtype=torch.long)
    out_channels = len(affiliation_encoder.classes_)

    # Create train, valid, test mask
    train_idx, test_idx = train_test_split(range(len(df)), test_size=0.2, random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=seed, shuffle=True)
    
    train_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask[test_idx] = True
    
        
    if os.path.isfile(savepath):
        data = torch.load(savepath)
    else:
        if not os.path.exists(savepath[:-7]):
            os.makedirs(savepath[:-7])
        author_id_to_index = {aid: idx for idx, aid in enumerate(df['author_id'].unique())}

        label_num = str(label_num)

        # Create edge index based on paper_id
        paper_to_authors = df.groupby('paper_id')['author_id'].apply(list).to_dict()
        edge_index = []
        for authors in paper_to_authors.values():
            if len(authors) > 1:
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        edge_index.append([author_id_to_index[authors[i]], author_id_to_index[authors[j]]])
                        edge_index.append([author_id_to_index[authors[j]], author_id_to_index[authors[i]]])


        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create Data object
        data = Data(x=tensor_data, edge_index=edge_index, y=author_labels)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        

        torch.save(data, savepath)

    return data, out_channels, affiliation_encoder, test_mask

if __name__=="__main__":
    data, out_channels, affiliation_encoder, test_mask = process_data(feature_dim=1000)