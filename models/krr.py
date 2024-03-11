import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from typing import Optional
from models.mlp import MLP

class KernelRidgeRegression_RBF(torch.nn.Module):
    def __init__(self, alpha: float, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x_train: Tensor, y_train: Tensor, x_test: Tensor) -> Tensor:
        K_train = self.kernel_function(x_train, x_train) + (torch.eye(x_train.shape[0]) * torch.tensor(self.alpha * y_train.shape[0])).cuda()
        K_test = self.kernel_function(x_test, x_train)
        
        inv_K_train = torch.inverse(K_train)
        alpha_hat = torch.matmul(inv_K_train, y_train)
        y_pred = torch.matmul(K_test, alpha_hat)

        return y_pred

    def kernel_function(self, X1: Tensor, X2: Tensor) -> Tensor:
        D = torch.cdist(X1, X2, p=2)
        K = torch.exp(-self.gamma * D**2)
        return K

class KernelRidgeRegression_MLP(torch.nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x_train: Tensor, y_train: Tensor, x_test: Tensor, mlp: MLP) -> Tensor:
        K_train = self.kernel_function(x_train, x_train, mlp) + (torch.eye(x_train.shape[0]) * torch.tensor(self.alpha * y_train.shape[0])).cuda()
        K_test = self.kernel_function(x_test, x_train, mlp)
        
        inv_K_train = torch.inverse(K_train)
        alpha_hat = torch.matmul(inv_K_train, y_train)
        y_pred = torch.matmul(K_test, alpha_hat)
        return y_pred

    def kernel_function(self, X1: Tensor, X2: Tensor, MLP: MLP) -> Tensor:
        mlp_x1 = MLP.forward(X1)
        mlp_x2 = MLP.forward(X2)
        K=torch.matmul(mlp_x1, mlp_x2.T)
        return K
    
import torch


class KernelRidgeRegression_NNGP(torch.nn.Module):
    def __init__(self, krr_size: int ,alpha: float, input_dim, hidden_dim):
        super(KernelRidgeRegression_NNGP, self).__init__()
        self.alpha = alpha
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights = []
        self.biases = []
        for i in range(krr_size):
            self.weights.append(nn.Parameter(torch.randn(input_dim, hidden_dim)).cuda())
            self.biases.append(nn.Parameter(torch.randn(hidden_dim)).cuda())
            
    def forward(self, x_train: Tensor, y_train: Tensor, x_test: Tensor, index) -> Tensor:
        K_train = self.kernel_function(x_train, x_train, index) + (torch.eye(x_train.shape[0]) * torch.tensor(self.alpha * y_train.shape[0])).cuda()
        K_test = self.kernel_function(x_test, x_train, index)
        
        inv_K_train = torch.inverse(K_train)
        alpha_hat = torch.matmul(inv_K_train, y_train)
        y_pred = torch.matmul(K_test, alpha_hat)

        return y_pred

    def kernel_function(self, x, x_prime, index):
        phi_x = []
        phi_x_prime = []
        for i in range(len(index)):
            phi_x.append(torch.relu(torch.matmul(x, self.weights[index[i]]) + self.biases[index[i]]))
            phi_x_prime.append(torch.relu(torch.matmul(x_prime, self.weights[index[i]]) + self.biases[index[i]]))
        phi_x = torch.concat(phi_x, dim=1)
        phi_x_prime = torch.concat(phi_x_prime, dim=1)
        inner_prod =  torch.matmul(phi_x, phi_x_prime.t()) / torch.sqrt(torch.tensor(len(index) * self.hidden_dim))

        return inner_prod