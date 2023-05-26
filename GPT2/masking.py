import torch
import torch.nn as nn


class PaddingMask(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset=0):
        '''
        :param x: input tensor (batch, seq_len)
        :param offset: In case of using PAST attention then it is used to pad the final concatenated k,v (during inference)
        '''
        # x: (batch, seq_len)
        is_pad = (x == self.pad_idx).unsqueeze(-2)  # (batch, 1, seq_len)
        # creating padding mask for past generations with all False
        shifted = torch.zeros(x.size()[:-1] + (1, offset,), dtype=torch.bool, device=x.device)
        # concatenating them into input mask as prefix
        mask = torch.cat((shifted, is_pad), dim=-1)  # (batch_len, 1, offset + seq_len)
        return mask.expand(x.shape + mask.shape[-1:])  # (batch_len, seq_len, seq_len + offset)


class FutureMasking(nn.Module):
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (batch_len, seq_len)
        seq_len = x.size(-1)

        # Create shifted upper triangular matrix.
        future = torch.ones((seq_len, seq_len + offset),
                            dtype=torch.bool, device=x.device)

        future = future.triu(diagonal=offset + 1)  # (seq_len, seq_len + offset)
        mask = future.view((1,) * (x.ndim - 1) + future.size())  # (1, seq_len, seq_len + offset)
        return mask.expand(x.shape + mask.shape[-1:])
