"""
MultiQuery Attention:
    - different heads share a single set of keys and values
    - different heads share different set of queries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    def __init__(self, dims, num_heads, dropout=0.1):
        super(MultiQueryAttention, self).__init__()
        assert dims % num_heads == 0, "Input dimensions must be divisible by total number of heads"
        self.dim = dims
        self.num_heads = num_heads
        self.key_dim = self.value_dim = dims // num_heads

        self.proj_q = nn.Linear(dims, dims)  # different heads share different set of queries
        self.proj_k = nn.Linear(dims, self.key_dim)
        self.proj_v = nn.Linear(dims, self.value_dim)

        self.linear = nn.Linear(dims, dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, past=None, mask=None):
        batch, q_len, _ = q.shape
        q = self.proj_q(q)  # (b, q_len, dims)
        q = q.view(batch, q_len, self.num_heads, self.key_dim)  # (b, q_len, num_heads, k_d)
        k = self.proj_k(k).unsqueeze(2)  # (b, k_l, 1, k_d)
        v = self.proj_v(v).unsqueeze(2)  # (b, v_l, 1, k_d)
        # note: k_l == v_l

        if past is not None:
            k = torch.cat([k, past[0].unsqueeze(2)], dim=1)  # (b, k_l + p_k_l, 1, k_d)
            v = torch.cat([v, past[1].unsqueeze(2)], dim=1)  # (b, v_l + p_v_l, 1, k_d)
            # note: k_l == v_l

        k = k.repeat(1, 1, self.num_heads, 1)  # (b, k_l, num_heads, d)
        v = v.repeat(1, 1, self.num_heads, 1)  # (b, v_l, num_heads, d)

        q = q.transpose(-2, -3)  # (b, num_heads, q_len, k_d)
        k = k.transpose(-2, -3)  # (b, num_heads, k_l, k_d)
        v = v.transpose(-2, -3)  # (b, num_heads, v_l, k_d)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (b, num_heads , q_l, k_l)

        if mask is not None:
            mask = mask.unsqueeze(-3)
            attn_scores += mask.type_as(attn_scores) * attn_scores.new_tensor(-1e5)

        attn_scores = F.softmax(attn_scores, dim=-1)  # (b, num_heads , q_l, k_l)

        attn_values = torch.matmul(attn_scores, v)  # (b, num_heads, q_l , k_d)
        attn_values = attn_values.contiguous().view(batch, q_len, self.num_heads * self.key_dim)  # (b, q_len, dims)
        attn_values = self.linear(attn_values)

        past_k = k[:, 0, :]  # (b, k_len, k_d)
        past_v = v[:, 0, :]  # (b, k_len, k_d)
        return attn_values, (past_k, past_v)


if __name__ == "__main__":
    dim = 56
    heads = 8
    q_ = torch.randn(32, 3, dim)
    k_ = torch.randn(32, 3, dim)
    v_ = torch.randn(32, 3, dim)
    k_d = dim // heads
    past_ = (torch.randn(32, 6, k_d), torch.randn(32, 6, k_d))
    attn = MultiQueryAttention(dim, heads)
    op, (prev_k, prev_v) = attn(q_, k_, v_, past=past_)
    print(op.shape, prev_k.shape, prev_v.shape)
