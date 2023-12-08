import torch
import numpy as np

import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
import wandb
import argparse

from model import GAT, HAN
import pandas as pd
import data_gat
import data_han

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
parser.add_argument('--max_epoch', default=4000, type=int)
parser.add_argument('--dataset_dir', default='dataset', type=str)
parser.add_argument('--patience_threshold', default=30, type=int)
parser.add_argument('--feature_dim', default=300, type=int)
parser.add_argument('--sample_type', default='undersample', type=str)
parser.add_argument('--dropout', default=0.6, type=float)
parser.add_argument('--data_aug', default='rand_edge', type=str)
parser.add_argument('--model', default='GAT', type=str)
parser.add_argument('--p',default=0.01, type=float, help='Probability for random edge')
args = parser.parse_args()

# wandb.init(project='DM_final', name = f"dropout_{args.dropout}_feat_{args.feature_dim}_seed_{args.seed}_lr_{args.lr}_dim_h_{args.dim_h}_heads_{args.heads}")
wandb.init(project='DM_gat', name = f"dropout_{args.dropout}_feat_{args.feature_dim}_seed_{args.seed}_lr_{args.lr}_dim_h_{args.dim_h}_heads_{args.heads}")
wandb.config.update(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)

data_dict={
    'GAT':data_gat,
    'HAN':data_han
}

# Initialize data and model
data, out_channels, affiliation_encoder, test_mask = data_dict[args.model].process_data(label_num=args.label_num, seed = args.seed, feature_dim=args.feature_dim,sample_type=args.sample_type, data_aug = args.data_aug)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channels = data.x.shape[1]
out_channels = len(np.unique(data.y.numpy()))  # Number of unique labels

if args.model=='GAT':
    model = GAT(dim_in=in_channels, dim_h=args.dim_h, out_channels=out_channels, heads=args.heads, dropout=args.dropout)
elif args.model=='HAN':
    model = HAN(dim_in=-1, dim_h=args.dim_h, dim_out=out_channels, data=data, dropout=args.dropout)

data, model =data.to(device), model.to(device)

# Optimzer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()


def top_k_accuracy(outputs, labels, k=5):
    # Get the top k predictions from the outputs
    topk_preds = torch.topk(outputs, k, dim=1).indices

    # Check if the true labels are in the top k predictions
    correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))

    # Calculate top k accuracy
    topk_acc = correct.sum().float() / labels.size(0)
    return topk_acc

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    mask = data.train_mask
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    pred = out.argmax(dim=1)
    correct =(pred[mask] == data.y[mask])
    acc = correct.sum() / mask.sum()

    top_acc_3 = top_k_accuracy(out[mask], data.y[mask], k=3)
    top_acc_5 = top_k_accuracy(out[mask], data.y[mask], k=5)
    return loss.item(), acc, top_acc_3, top_acc_5

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out[mask], data.y[mask]).item()
        pred = out.argmax(dim=1)
        acc = (pred[mask] == data.y[mask]).sum() / mask.sum()
        
        top_acc_3 = top_k_accuracy(out[mask], data.y[mask], k=3)
        top_acc_5 = top_k_accuracy(out[mask], data.y[mask], k=5)
    return loss, acc, top_acc_3, top_acc_5


def visualize_top_k_predictions(mask, filename, k=2):
    model.eval()
    predictions_data = []

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        predictions = torch.topk(out, k, dim=1)

    for idx in torch.where(mask)[0][:5]:  # Visualize for the first 5 authors in the mask
        # Get the ground truth affiliation for the author
        true_affiliation_index = data.y[idx].item()
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
        true_affiliation_index = data.y[idx].item()
        true_affiliation = affiliation_encoder.inverse_transform([true_affiliation_index])[0]
        top_k_predictions = [(affiliation_encoder.inverse_transform([i.item()])[0], v.item()) for i, v in zip(predictions.indices[idx], predictions.values[idx])]
        predictions_data.append([idx.item(), true_affiliation, top_k_predictions])
    df = pd.DataFrame(predictions_data, columns=["author_id", "ground_truth_affiliation", "top_k_predictions"])
    df.to_csv(filename, index=False)




# Variables to track the best validation metrics
best_val_acc = 0.0
best_val_loss = float('inf')
patience, patience_threshold = 0, args.max_epoch # no early stopping for now

# current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

BEST_MODEL_PATH = f"./best_model/dropout_{args.dropout}_feat_{args.feature_dim}_seed_{args.seed}_lr_{args.lr}_dim_h_{args.dim_h}_heads_{args.heads}"

for epoch in tqdm(range(args.max_epoch), desc="Training Epochs"):
    loss, acc, top_acc_3, top_acc_5 = train()
    val_loss, val_acc, val_top_acc_3, val_top_acc_5 = evaluate(data.val_mask)
    
        
    # Update best metrics and log them if improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        wandb.log({"best_val_acc": best_val_acc})
        torch.save(model.state_dict(), BEST_MODEL_PATH)  # Save the model

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
    formatted_top_acc_3 = f"{top_acc_3:.4f}"
    formatted_top_acc_5 = f"{top_acc_5:.4f}"
    formatted_val_loss = f"{val_loss:.4f}"
    formatted_val_acc = f"{val_acc:.4f}"
    formatted_val_top_acc_3 = f"{val_top_acc_3:.4f}"
    formatted_val_top_acc_5 = f"{val_top_acc_5:.4f}"
    
    wandb.log({
        "train_loss":loss,
        "train_acc":acc,
        "train_top_acc_3":top_acc_3,
        "train_top_acc_5":top_acc_5,
        "val_loss":val_loss,
        "val_acc":val_acc,
        "val_top_acc_3":val_top_acc_3,
        "val_top_acc_5":val_top_acc_5,
    })
    
    print(f'Epoch: {epoch}, Acc: {formatted_acc}, Loss: {formatted_loss}, Val Loss: {formatted_val_loss}, Val Acc: {formatted_val_acc}')

model.load_state_dict(torch.load(BEST_MODEL_PATH))

test_loss, test_acc, test_top_acc_3, test_top_acc_5 = evaluate(data.test_mask)
formatted_loss = f"{test_loss:.4f}"
formatted_acc = f"{test_acc:.4f}"
formatted_top_acc_3 = f"{test_top_acc_3:.4f}"
formatted_top_acc_5 = f"{test_top_acc_5:.4f}"

wandb.log({
    "test_loss":test_loss,
    "test_acc":test_acc,
    "test_top_acc_3":test_top_acc_3,
    "test_top_acc_5":test_top_acc_5,
})

print(f'Test Loss: {formatted_loss}, Test Accuracy: {formatted_acc}, TOP-3: {formatted_top_acc_3}, TOP-5: {formatted_top_acc_5}')


# visualize_top_k_predictions(data['author'].test_mask, k=args.top_k)
filename = f"./preds/dropout_{args.dropout}_feat_{args.feature_dim}_seed_{args.seed}_lr_{args.lr}_dim_h_{args.dim_h}_heads_{args.heads}_pred.csv"
visualize_top_k_predictions(data.test_mask, k=args.top_k, filename=filename)
