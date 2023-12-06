import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def process_data(label_num=2, max_features=100, seed=0):

    label_num = str(label_num)
    papers_df = pd.read_csv(f'./dataset/{label_num}_processed_papers.csv')
    authors_df = pd.read_csv(f'./dataset/{label_num}_processed_authors.csv')

    # One-hot encoding for paper IDs
    paper_id_encoder = OneHotEncoder(sparse=False)
    paper_ids_onehot = paper_id_encoder.fit_transform(papers_df[['paper_id']])

    # One-hot encoding for author IDs
    author_id_encoder = OneHotEncoder(sparse=False)
    author_ids_onehot = author_id_encoder.fit_transform(authors_df[['author_id']])

    # Initialize HeteroData
    data = HeteroData()

    '''Create paper feature'''
    # TF-IDF encoding for paper text
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    # papers_df['combined_text'] = papers_df['paper_title'] + ' ' + papers_df['paper_area'] + ' ' + papers_df['paper_abstract']
    papers_df['combined_text'] = papers_df['paper_area'] + ' ' + papers_df['paper_abstract']+ ' ' + papers_df['name']+ ' ' + str(papers_df['year'])
    tfidf_embeddings = tfidf_vectorizer.fit_transform(papers_df['combined_text']).toarray()
    data['paper'].x = torch.tensor(tfidf_embeddings, dtype=torch.float)

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