import torch
import numpy as np

import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
import wandb
import argparse

from model import HANModel
# from data_tfidf import process_data
# from data_glove import process_data
import data_tfidf
import data_glove
import torch.nn as nn

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--top_k', default=3, type=int)
parser.add_argument('--dim_h', default=512, type=int)
parser.add_argument('--heads', default=8, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--label_num', default=2, type=int)
parser.add_argument('--max_features', default=100, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--max_epoch', default=100, type=int)
parser.add_argument('--data_type', default='tfidf', type=str)
args = parser.parse_args()

data_dict={
    'tfidf':data_tfidf.process_data,
    'glove':data_glove.process_data,
}

wandb.init(project='DM_final')
wandb.config.update(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)

# Initialize data and model
data, out_channels, affiliation_encoder, test_mask = data_dict[args.data_type](label_num=args.label_num, max_features = args.max_features, seed = args.seed)
model = HANModel(dim_in=-1, dim_h=args.dim_h, dim_out=out_channels, data=data)

# Optimzer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)# ['author'][data['author'].train_mask]
    mask = data['author'].train_mask
    loss = criterion(out[mask], data['author'].y[mask])
    loss.backward()
    optimizer.step()
    pred = out.argmax(dim=1)
    acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
    return loss.item(), acc

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)# ['author'][mask]
        loss = criterion(out[mask], data['author'].y[mask]).item()
        pred = out.argmax(dim=1)
        acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
    return loss, acc

def visualize_top_k_predictions(mask, k=2):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)# ['author'][mask]
        predictions = torch.topk(out, k, dim=1)

    for idx in torch.where(mask)[0][:5]:  # Visualize for the first 5 authors in the mask
        print(f"Author {idx.item()}:")
        for i, v in zip(predictions.indices[idx], predictions.values[idx]):
            print(f"  Affiliation: {affiliation_encoder.inverse_transform([i.item()])[0]}, Score: {v.item()}")

# Variables to track the best validation metrics
best_val_acc = 0.0
best_val_loss = float('inf')
patience, patience_threshold = 0, 10  

for epoch in tqdm(range(args.max_epoch), desc="Training Epochs"):
    loss, acc = train()
    val_loss, val_acc = evaluate(data['author'].val_mask)
    
        
    # Update best metrics and log them if improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        wandb.log({"best_val_acc": best_val_acc})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wandb.log({"best_val_loss": best_val_loss})
        patience = 0  # Reset patience since we have improvement
    else:
        patience += 1

    # Early stopping check
    if patience >= patience_threshold:
        print(f"Threshold : {patience_threshold}, Early stopping triggered")
        break
    
    formatted_loss = f"{loss:.4f}"
    formatted_acc = f"{acc:.4f}"
    formatted_val_loss = f"{val_loss:.4f}"
    formatted_val_acc = f"{val_acc:.4f}"
    
    wandb.log({
        "train_loss":formatted_loss,
        "train_acc":formatted_acc,
        "val_loss":formatted_val_loss,
        "val_acc":formatted_val_acc
    })
    
    print(f'Epoch: {epoch}, Acc: {formatted_acc}, Loss: {formatted_loss}, Val Loss: {formatted_val_loss}, Val Acc: {formatted_val_acc}')

test_loss, test_acc = evaluate(data['author'].test_mask)
formatted_loss = f"{test_loss:.4f}"
formatted_acc = f"{test_acc:.4f}"

wandb.log({
    "test_loss":formatted_loss,
    "test_acc":formatted_acc,
})

print(f'Test Loss: {formatted_loss}, Test Accuracy: {formatted_acc}')


visualize_top_k_predictions(data['author'].test_mask, k=args.top_k)