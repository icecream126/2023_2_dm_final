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
from data import *
import torch.nn as nn
import os
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--top_k', default=4, type=int)
parser.add_argument('--dim_h', default=512, type=int)
parser.add_argument('--heads', default=8, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--label_num', default=10, type=int)
parser.add_argument('--max_features', default=100, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--max_epoch', default=3000, type=int)
parser.add_argument('--dataset_dir', default='dataset', type=str)
parser.add_argument('--patience_threshold', default=30, type=int)
parser.add_argument('--feature_dim', default=300, type=int)
parser.add_argument('--sample_type', default='undersample', type=str)
args = parser.parse_args()

wandb.init(project='DM_final', name = f"sample_{args.sample_type}_feat_{args.feature_dim}_seed_{args.seed}_lr_{args.lr}_dim_h_{args.dim_h}_heads_{args.heads}")
wandb.config.update(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)

# Initialize data and model
data, out_channels, affiliation_encoder, test_mask = process_data(label_num=args.label_num, seed = args.seed, feature_dim=args.feature_dim,sample_type=args.sample_type)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HANModel(dim_in=-1, dim_h=args.dim_h, dim_out=out_channels, data=data)
data, model =data.to(device), model.to(device)

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


def visualize_top_k_predictions(mask, filename, k=2):
    model.eval()
    predictions_data = []

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        predictions = torch.topk(out, k, dim=1)

    for idx in torch.where(mask)[0][:5]:  # Visualize for the first 5 authors in the mask
        # Get the ground truth affiliation for the author
        true_affiliation_index = data['author'].y[idx].item()
        true_affiliation = affiliation_encoder.inverse_transform([true_affiliation_index])[0]

        # Generate a string of predictions with scores
        predictions_str = ', '.join([f"{affiliation_encoder.inverse_transform([i.item()])[0]} ({v.item():.2f})" for i, v in zip(predictions.indices[idx], predictions.values[idx])])

        # Add both ground truth and predictions to the data
        predictions_data.append([idx.item(), true_affiliation, predictions_str])

    # Log to wandb
    columns = ["author_id", "ground_truth_affiliation", "predictions"]
    wandb_table = wandb.Table(columns=columns, data=predictions_data)
    wandb.log({"top_k_predictions": wandb_table})
    
    predictions_data = []
    for idx in torch.where(mask)[0]:
        true_affiliation_index = data['author'].y[idx].item()
        true_affiliation = affiliation_encoder.inverse_transform([true_affiliation_index])[0]
        top_k_predictions = [(affiliation_encoder.inverse_transform([i.item()])[0], v.item()) for i, v in zip(predictions.indices[idx], predictions.values[idx])]
        predictions_data.append([idx.item(), true_affiliation, top_k_predictions])
    df = pd.DataFrame(predictions_data, columns=["author_id", "ground_truth_affiliation", "top_k_predictions"])
    df.to_csv(filename, index=False)




# Variables to track the best validation metrics
best_val_acc = 0.0
best_val_loss = float('inf')
patience, patience_threshold = 0, args.max_epoch # no early stopping for now

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
        "train_loss":loss,
        "train_acc":acc,
        "val_loss":val_loss,
        "val_acc":val_acc
    })
    
    print(f'Epoch: {epoch}, Acc: {formatted_acc}, Loss: {formatted_loss}, Val Loss: {formatted_val_loss}, Val Acc: {formatted_val_acc}')

test_loss, test_acc = evaluate(data['author'].test_mask)
formatted_loss = f"{test_loss:.4f}"
formatted_acc = f"{test_acc:.4f}"

wandb.log({
    "test_loss":test_loss,
    "test_acc":test_acc,
})

print(f'Test Loss: {formatted_loss}, Test Accuracy: {formatted_acc}')


# visualize_top_k_predictions(data['author'].test_mask, k=args.top_k)
filename = f"./preds/sample_{args.sample_type}_feat_{args.feature_dim}_seed_{args.seed}_lr_{args.lr}_dim_h_{args.dim_h}_heads_{args.heads}_pred.csv"
visualize_top_k_predictions(data['author'].test_mask, k=args.top_k, filename=filename)
