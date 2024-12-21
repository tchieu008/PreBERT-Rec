import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import save_npz, load_npz
import os
import pickle

# Configuration
class Args:
    epochs = 10
    learning_rate = 0.01
    reg = 0.01
    batch_size = 1024

args = Args()

class FactorizationMachine:
    def __init__(self, n_factors, n_features, feature_names):
        self.n_factors = n_factors
        self.w0 = 0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(scale=0.01, size=(n_features, n_factors))
        self.feature_names = feature_names
    
    def predict(self, X):
        if isinstance(X, csr_matrix):
            X = X.toarray() 
        linear_terms = np.dot(X, self.w) + self.w0
        interactions = 0.5 * np.sum(
            np.power(np.dot(X, self.V), 2) - np.dot(np.power(X, 2), np.power(self.V, 2)),
            axis=1
        )
        return (linear_terms + interactions).flatten()
    
    def fit(self, X, y, epochs, learning_rate, reg, checkpoint_path, batch_size):
        m, n = X.shape
        total_steps = epochs * (m // batch_size + (m % batch_size > 0))
        with tqdm(total=total_steps, desc="Training Progress") as pbar:
            for epoch in range(epochs):
                epoch_loss = 0
                for i in range(0, m, batch_size):
                    X_batch = X[i:i+batch_size]
                    y_batch = y[i:i+batch_size]
                    
                    for xi, yi in zip(X_batch, y_batch):
                        xi = xi.toarray().flatten() if isinstance(xi, csr_matrix) else xi.flatten()
                        pred = self.predict(xi.reshape(1, -1))[0]
                        error = yi - pred
                        epoch_loss += error**2
                        
                        self.w0 += learning_rate * (error - reg * self.w0)
                        self.w += learning_rate * (error * xi - reg * self.w)
                        for f in range(self.n_factors):
                            self.V[:, f] += learning_rate * (
                                error * (xi * np.dot(xi, self.V[:, f]) - self.V[:, f] * xi**2) - reg * self.V[:, f]
                            )
                    pbar.update(1)
                epoch_loss /= m
                print(f"Epoch {epoch + 1}/{epochs} completed, Loss: {epoch_loss:.4f}")
            # Save checkpoint after each epoch
            self.save_checkpoint(checkpoint_path)

    def get_embedding(self, feature_name):
        try:
            index = np.where(self.feature_names == feature_name)[0][0]
            return self.V[index]
        except IndexError:
            print(f"Feature '{feature_name}' not found.")
            return None

    def save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'w0': self.w0,
            'w': self.w,
            'V': self.V,
            'feature_names': self.feature_names
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            self.w0 = checkpoint['w0']
            self.w = checkpoint['w']
            self.V = checkpoint['V']
            self.feature_names = checkpoint['feature_names']
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

def run(file_path, n_factors, checkpoint_path, sparse_matrix_path):
    df = pd.read_csv(file_path)
    df = df[['overall', 'reviewerID', 'itemID']]
    
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(df[['reviewerID', 'itemID']])

    if not os.path.exists(sparse_matrix_path):
        save_npz(sparse_matrix_path, encoded_features)
    else:
        encoded_features = load_npz(sparse_matrix_path)

    X = encoded_features.astype(np.float64)
    y = df['overall'].apply(lambda x: 1 if x >= 4 else -1).values


    feature_names = encoder.get_feature_names_out(['reviewerID', 'itemID'])
    n_features = X.shape[1]
    fm = FactorizationMachine(n_factors, n_features, feature_names)

    if os.path.exists(checkpoint_path):
        fm.load_checkpoint(checkpoint_path)
    else:
        fm.fit(X, y, epochs=args.epochs, learning_rate=args.learning_rate, reg=args.reg, checkpoint_path=checkpoint_path, batch_size=args.batch_size)
    return fm
