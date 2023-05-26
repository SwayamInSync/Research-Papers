import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embeds, max_len, device=torch.device('cpu')):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embeds
        self.max_len = max_len
        self.device = device

    def forward(self, x):
        encoding = torch.zeros(self.max_len, self.embed_size, device=self.device)
        # encoding.requires_grad = False
        pos = torch.arange(0, self.max_len, device=self.device)
        pos = pos.float().unsqueeze(dim=1)
        i = torch.arange(0, self.embed_size, step=2, device=self.device).float()

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (i / self.embed_size)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (i / self.embed_size)))

        batch_size, seq_len = x.size()
        return encoding[:seq_len, :].expand(batch_size, seq_len, self.embed_size)


class PositionalEmbedding(nn.Embedding):
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        position = torch.arange(offset, offset + x.size(-1),
                                dtype=torch.long, device=x.device)
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x)

        return super().forward(position)


class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, dims):
        super(WordEmbeddings, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, dims)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding_layer(x)  # (batch, seq_len, dims)
        return x
