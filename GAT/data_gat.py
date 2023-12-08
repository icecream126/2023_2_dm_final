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
    
    savepath = f'./dataset/graph/{feature_dim}/{sample_type}/10_label/data.pt'
    df = pd.read_csv(f'./dataset/merged/data_{feature_dim}.csv')
    
    df_x = df[[col for col in df.columns if 'paper_abstract_filtered' in col]]
    df_y = df['author_affiliation']
    df = pd.concat([df_x, df_y], axis=1)
    
    
    # label encoding
    le = LabelEncoder()
    le.fit(df['author_affiliation'].unique())
    df['y'] = le.transform(df['author_affiliation'])
    
    out_channels = len(df['y'].unique())
    
    # Create train, valid, test mask
    train_idx, test_idx = train_test_split(range(len(df)), test_size=0.2, random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=seed, shuffle=True)
    
    train_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask[test_idx] = True
    
    data=torch.load(savepath)
    

    return data, out_channels, le, test_mask