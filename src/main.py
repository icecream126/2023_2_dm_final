import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, HANConv
import torch.nn.functional as F
import torch.nn as nn
import torch

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm

import torch.nn as nn
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from data import process_data
from model import HeteroGNN
import torch.backends.cudnn as cudnn
import random

# args.seed
# args.

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

data, out_channels, affiliation_encoder = process_data()

model = HeteroGNN(hidden_channels=128, out_channels=out_channels)  # num_classes is the number of possible affiliations
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# # Training loop
# for epoch in tqdm(range(10), desc="Training Epochs"):  # number of epochs
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x_dict, data.edge_index_dict)
#     loss = criterion(out['author'][data['author'].train_mask], data['author'].y[data['author'].train_mask])
#     loss.backward()
#     optimizer.step()

#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item()}')

for epoch in tqdm(range(10), desc="Training Epochs"):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = criterion(out['author'], batch['author'].y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f'Epoch {epoch}, Loss: {avg_train_loss}')
    
    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            out = model(batch.x_dict, batch.edge_index_dict)
            valid_loss = criterion(out['author'], batch['author'].y)
            total_valid_loss += valid_loss.item()
    
    avg_valid_loss = total_valid_loss / len(valid_loader)
    print(f'Epoch {epoch}, Validation Loss: {avg_valid_loss}')
    

# Prediction
model.eval()
all_preds = []
total_test_loss=0
with torch.no_grad():
    for batch in test_loader:
        out = model(batch.x_dict, batch.edge_index_dict)['author']
        all_preds.append(out)
        test_loss = criterion(out['author'], batch['author'].y)
        total_test_loss += test_loss.item()
        
avg_test_loss = total_test_loss / len(test_loader)
all_preds = torch.cat(all_preds, dim=0)


k=4
top_k_preds = torch.topk(out, k, dim=1)   
# top_k_preds will contain two tensors: scores and indices
print('score : ',top_k_preds[0]) # contains the scores
print('pred : ', top_k_preds[1]) # contains the indices of the top k predictions

# If you want to convert these indices back to labels (assuming you have a label encoder)
label_encoder = affiliation_encoder  # Your label encoder
top_k_labels = label_encoder.inverse_transform(top_k_preds[1])
print('top_k_labels : ', top_k_labels)