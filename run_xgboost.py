import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

def top_k_accuracy(y_true, y_prob, k=5):
    # Get indices of the top k probabilities
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]

    # Check if the true labels are in the top k predictions
    matches = np.any(top_k_preds == y_true.reshape((-1, 1)), axis=1)

    # Calculate top k accuracy
    topk_acc = np.mean(matches)
    return topk_acc

log_df = pd.read_csv('./dataset/merged/data_1000.csv', encoding_errors='ignore')
df_x = log_df[[col for col in log_df.columns if 'paper_abstract_filtered' in col]]
df_y = log_df['author_affiliation_type']
df = pd.concat([df_x, df_y], axis=1)

# label encoding
le = LabelEncoder()
le.fit(df['author_affiliation_type'].unique())
df['y'] = le.transform(df['author_affiliation_type'])

X = df[[col for col in df.columns if 'paper_abstract_filtered' in col]]
y = df['y']

rus = RandomOverSampler()
X_res, y_res = rus.fit_resample(X, y)

# Feature selection (variance threshold)
v_threshold = VarianceThreshold(threshold=0.001)
v_threshold.fit(X_res)
columns_vari = X_res.columns[v_threshold.get_support()]
# print(len(columns_vari)/len(X_res.columns))

X_train, X_test, y_train, y_test = train_test_split(X_res[columns_vari], y_res, test_size=0.2, shuffle=True, random_state=args.seed)
abc = XGBClassifier()
model = abc.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Top-1 Accuracy:", accuracy)

# Calculate top-k accuracy
top_3_acc = top_k_accuracy(y_test.values, y_prob, k=3)
top_5_acc = top_k_accuracy(y_test.values, y_prob, k=5)

print("Top-3 Accuracy:", top_3_acc)
print("Top-5 Accuracy:", top_5_acc)