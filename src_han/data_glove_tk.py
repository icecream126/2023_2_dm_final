import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import gensim.downloader as api
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from gensim.models import KeyedVectors

def load_glove_model(glove_file_path):
    glove_model = api.load("glove-wiki-gigaword-100")  # 100-dimensional GloVe model
    return glove_model

def get_paper_embedding(tokenized_abstract, glove_model, embedding_size):
    embeddings = [glove_model[word] for word in tokenized_abstract if word in glove_model]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(embedding_size)

def process_data(dataset_dir='dataset',label_num=2, max_features=100, seed=0, glove_file_path='path_to_glove_vectors.txt'):
    
    label_num = str(label_num)
    papers_df = pd.read_csv(f'./{dataset_dir}/{label_num}_processed_papers.csv')
    authors_df = pd.read_csv(f'./{dataset_dir}/{label_num}_processed_authors.csv')
    
    glove_model = load_glove_model(glove_file_path)
    embedding_size = glove_model.vector_size

    data = HeteroData()

    '''Create paper feature'''
    paper_embeddings = np.array([get_paper_embedding(tokenized_abstract, glove_model, embedding_size) for tokenized_abstract in papers_df['paper_abstract_tk']])

    data['paper'].x = torch.tensor(paper_embeddings, dtype=torch.float)


    '''Create author feature'''
    # One-hot encoding for author position and affiliation type
    position_encoder = OneHotEncoder(sparse=False)
    positions_onehot = position_encoder.fit_transform(authors_df[['author_position']])
    affiliation_type_encoder = OneHotEncoder(sparse=False)
    affiliation_type_onehot = affiliation_type_encoder.fit_transform(authors_df[['author_affiliation_type']])
    author_features = np.hstack((positions_onehot, affiliation_type_onehot))
    data['author'].x = torch.tensor(author_features, dtype=torch.float)

    '''Create author label'''
    # Add target labels for authors (affiliation)
    affiliation_encoder = LabelEncoder()
    authors_df['affiliation_encoded'] = affiliation_encoder.fit_transform(authors_df['author_affiliation']) # [54819]
    author_labels = torch.tensor(authors_df['affiliation_encoded'].values, dtype=torch.long) # [54819]
    data['author'].y = author_labels
    out_channels = len(affiliation_encoder.classes_)

    '''Create train, valid, test mask'''
    train_idx, test_idx = train_test_split(range(len(authors_df)), test_size=0.2, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=seed)

    train_mask = torch.zeros(len(authors_df), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(authors_df), dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(len(authors_df), dtype=torch.bool)
    test_mask[test_idx] = True

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

    data['author', 'to', 'paper'].edge_index = edge_index
    data['paper', 'to', 'author'].edge_index = torch.flip(edge_index, [0])
    

    return data, out_channels, affiliation_encoder, test_mask