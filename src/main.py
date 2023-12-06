import torch
import numpy as np
import torch.nn as nn

import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm

from model import HANModel
from data import process_data

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

data, out_channels, affiliation_encoder, test_mask = process_data()
# Initialize the model
model = HANModel(paper_in_channels=data['paper'].x.shape[1], author_in_channels=data['author'].x.shape[1], hidden_channels=128, out_channels=out_channels)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)['author'][data['author'].train_mask]
    loss = criterion(out, data['author'].y[data['author'].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)['author'][mask]
        loss = criterion(out, data['author'].y[mask]).item()
        pred = out.argmax(dim=1)
        correct = pred.eq(data['author'].y[mask]).sum().item()
        acc = correct / mask.sum().item()
    return loss, acc

def visualize_top_k_predictions(mask, k=5):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)['author'][mask]
        predictions = torch.topk(out, k, dim=1)

    for idx in torch.where(mask)[0][:5]:  # Visualize for the first 5 authors in the mask
        print(f"Author {idx.item()}:")
        for i, v in zip(predictions.indices[idx], predictions.values[idx]):
            print(f"  Affiliation: {affiliation_encoder.inverse_transform([i.item()])[0]}, Score: {v.item()}")


for epoch in tqdm(range(10), desc="Training Epochs"):
    loss = train()
    val_loss, val_acc = evaluate(data['author'].val_mask)
    print(f'Epoch: {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

test_loss, test_acc = evaluate(data['author'].test_mask)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')


visualize_top_k_predictions(data['author'].test_mask, k=5)