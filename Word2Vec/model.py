import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Word2Vec, self).__init__()
        self.num_embedding = num_embeddings
        self.e = nn.Embedding(num_embeddings, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, x):
        x = self.e(x)
        x = self.fc(x)
        return x
