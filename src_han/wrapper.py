import torch
from sklearn.base import BaseEstimator

class PyTorchClassifierWrapper(BaseEstimator):
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, X, y):
        # Convert X and y to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Training loop (simplified for illustration)
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(X_tensor)
        loss = self.criterion(output, y_tensor)
        loss.backward()
        self.optimizer.step()

    def predict(self, X):
        # Convert X to a PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            _, predicted = torch.max(output, 1)
        return predicted.numpy()