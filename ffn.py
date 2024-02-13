'''target model. predicts localization of PROKARYOTIC proteins'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt



# recieves output of transformer as input, clone of 
class FFN(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 6, hidden_dim: int = 32,
                 n_hidden_layers: int = 4, dropout: float = 0.25, batch_size: int = 32):
        """
        Simple Feed forward model with default parameters like the network tha is ued in the SeqVec paper.
        Args:
            embeddings_dim: dimension of the input
            hidden_dim: dimension of the hidden layers
            output_dim: output dimension (number of classes that should be classified)
            n_hidden_layers: number of hidden layers (0 by default)
            dropout: dropout ratio of every layer
        """
        super(FFN, self).__init__()
        self.embeddings_dim = embeddings_dim
        self.n_hidden_layers = n_hidden_layers

        self.input = nn.Sequential(
            nn.Linear(in_features=embeddings_dim, out_features=hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_dim)
        )
       
        self.hidden = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.hidden.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=hidden_dim)
            ))
        self.output = nn.Linear(hidden_dim, output_dim)

    #__init__

    def forward(self, x, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] embedding tensor 

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        x = x.squeeze()
        x = x.float() 

        o = self.input(x)
        for hidden_layer in self.hidden:
            o = hidden_layer(o)
        return self.output(o)
