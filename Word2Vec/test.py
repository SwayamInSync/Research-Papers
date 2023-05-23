from sklearn.manifold import TSNE
from re import T
from MulticoreTSNE import MulticoreTSNE as TSNE
from model import Word2Vec
import torch
from dataloader import get_word_loader
from matplotlib import pyplot as plt
import numpy as np
import yellowbrick

path = "/home/mia/swayam/exp/exp2/checkpoints/10000_4.137189774715807.pt"
dataset, dataloader = get_word_loader(batch_size=512, shuffle=True)
vocab = dataset.vocab
checkpoint = torch.load(path)
model = Word2Vec(len(vocab), 300)
model.load_state_dict(checkpoint["state_dict"]())
indices = torch.tensor(list(vocab.stoi.values()))
embeds = model.e(indices).detach()
print(embeds.shape)

tsne = TSNE(n_jobs=4)
Y = tsne.fit_transform(embeds)
x = Y[:, 0]
y = Y[:, 1]

yellowbrick.text.tsne.tsne(x, y, show=True)
