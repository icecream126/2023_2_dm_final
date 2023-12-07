import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from torch_geometric.transforms import VirtualNode

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import os

def process_data(label_num=10, seed=0, feature_dim=500, sample_type='undersample', data_aug=None, p=0.01):
    
    savepath = f'./dataset/graph/{sample_type}/{label_num}_label/data.pt'
    df = pd.read_csv(f'./dataset/merged/data_{feature_dim}.csv')
    
    # # Create paper feature
    # filtered_columns = [col for col in df.columns if col.startswith('paper_abstract_filtered_')]
    # filtered_data = df[filtered_columns]
    # tensor_data = torch.tensor(filtered_data.values, dtype=torch.float32)

    # # Create author label
    # affiliation_encoder = LabelEncoder()
    # df['affiliation_encoded'] = affiliation_encoder.fit_transform(df['author_affiliation']) 
    # author_labels = torch.tensor(df['affiliation_encoded'].values, dtype=torch.long)
    # out_channels = len(affiliation_encoder.classes_)
    
    df_x = df[[col for col in df.columns if 'paper_abstract_filtered' in col]]
    df_y = df['author_affiliation']
    df = pd.concat([df_x, df_y], axis=1)
    
    # label encoding
    le = LabelEncoder()
    le.fit(df['author_affiliation'].unique())
    df['y'] = le.transform(df['author_affiliation'])
    
    X = df[[col for col in df.columns if 'paper_abstract_filtered' in col]]
    y = df['y']
    out_channels = len(y.unique())
    
    rus = RandomOverSampler()
    X_res, y_res = rus.fit_resample(X, y)
    # Feature selection (variance threshold)
    v_threshold = VarianceThreshold(threshold=0.001)
    v_threshold.fit(X_res)
    columns_vari = X_res.columns[v_threshold.get_support()]
    print(len(columns_vari)/len(X_res.columns))
    
    X_res_chosen = X_res[columns_vari]
    tensor_data = torch.tensor(X_res_chosen.values, dtype=torch.float32)
    author_labels = torch.tensor(y_res.values, dtype=torch.long)
    
    
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
        
        virtual_node_transform = VirtualNode(num_classes=out_channels)
        data = virtual_node_transform(data)

        torch.save(data, savepath)

    return data, out_channels, le, test_mask

if __name__=="__main__":
    data, out_channels, affiliation_encoder, test_mask = process_data()
