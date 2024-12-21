import os
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import torch

class SVD():
    def __init__(self, data_path, num_factors):
        self.df = pd.read_csv(data_path)
        self.users = sorted(self.df['reviewerID'].unique())
        self.items = sorted(self.df['itemID'].unique())

        self.users_id_dict = {u: index for index, u in enumerate(self.users)}
        self.items_id_dict = {i: index for index, i in enumerate(self.items)}

        self.rows = []
        self.cols = []
        self.data = []

        self.beta = 0.9
        self.lmbda = 0.0002
        self.k = num_factors
        self.learning_rate = 0.01
        self.iterations = 1000
        self.u_dim = len(self.users)
        self.i_dim = len(self.items)

        self._init_ratings_matrix()

    def _init_ratings_matrix(self):
        for item in self.df.itertuples():
            r = item[3]
            u = item[1]
            i = item[2]
            iu = self.users_id_dict[u]
            ii = self.items_id_dict[i]
            self.rows.append(iu)
            self.cols.append(ii)
            self.data.append(r)
        
        ratings = np.zeros((self.u_dim, self.i_dim))
        for r, c, d in zip(self.rows, self.cols, self.data):
            ratings[r, c] = d

        self.ratings = ratings
        self.sparse_ratings = self._create_sparse_matrix(self.data, self.u_dim, self.i_dim)

    def _create_sparse_matrix(self, data, len_user, len_item):
        return sparse.csc_matrix((data, (self.rows, self.cols)), shape=(len_user, len_item))

    def _create_embeddings(self, n):
        return 6 * np.random.random((n, self.k)) / self.k

    def _predict(self, emb_user, emb_item):
        return np.dot(emb_user, emb_item.T)

    def _cost(self, emb_user, emb_item):
        p_predict = self._predict(emb_user, emb_item)
        p_data = [p_predict[r][c] for r, c in zip(self.rows, self.cols)]
        predicted = self._create_sparse_matrix(p_data, emb_user.shape[0], emb_item.shape[0])
        return np.sum((self.sparse_ratings - predicted).power(2)) / len(self.data)

    def _gradient(self, emb_user, emb_item):
        p_predict = self._predict(emb_user, emb_item)
        p_data = [p_predict[r][c] for r, c in zip(self.rows, self.cols)]
        sparse_predicted = self._create_sparse_matrix(p_data, emb_user.shape[0], emb_item.shape[0])
        delta = self.sparse_ratings - sparse_predicted

        grad_user = (-2 / self.df.shape[0]) * (delta @ emb_item) + 2 * self.lmbda * emb_user
        grad_item = (-2 / self.df.shape[0]) * (delta.T @ emb_user) + 2 * self.lmbda * emb_item
        return grad_user, grad_item

    def train(self):
        emb_user = self._create_embeddings(self.u_dim)
        emb_item = self._create_embeddings(self.i_dim)
        v_user = np.zeros_like(emb_user)
        v_item = np.zeros_like(emb_item)

        for i in range(self.iterations):
            grad_user, grad_item = self._gradient(emb_user, emb_item)
            v_user = self.beta * v_user + (1 - self.beta) * grad_user 
            v_item = self.beta * v_item + (1 - self.beta) * grad_item
            emb_user -= self.learning_rate * v_user
            emb_item -= self.learning_rate * v_item

            if (i + 1) % 50 == 0:
                print(f"\nIteration {i + 1}:")

        self.emb_user = emb_user
        self.emb_item = emb_item

    def get_embeddings(self):
        if hasattr(self, 'emb_user'):
            return self.emb_user, self.emb_item
        else:
            raise Exception('Please train the model first.')

    def get_user_embedding(self, user_id):
        index = self.users_id_dict[user_id]
        return self.emb_user[index, :]

    def get_item_embedding(self, item_id):
        index = self.items_id_dict[item_id]
        return self.emb_item[index, :]

def initialize_svd(data_path, num_factors, checkpoint_path='model/DeepCGSR/chkpt/svd.pt'):
    if os.path.exists(checkpoint_path):
        svd = torch.load(checkpoint_path)
    else:
        svd = SVD(data_path, num_factors)
        svd.train()
        torch.save(svd, checkpoint_path)
    return svd
