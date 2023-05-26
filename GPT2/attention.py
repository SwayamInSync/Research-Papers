import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dims, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.proj_q = nn.Linear(dims, dims)
        self.proj_k = nn.Linear(dims, dims)
        self.proj_v = nn.Linear(dims, dims)
        self.linear = nn.Linear(dims, dims)

        self.num_heads = heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, past=None, mask=None):
        q = self.proj_q(q)  # (batch, q_len, dim)
        k = self.proj_k(k)  # (batch, q_len, dim)
        v = self.proj_v(v)  # (batch, q_len, dim)

        # K,V coming from another block
        # we can concatenate them, since the increase in length will be handled while matrix multiplication in attention
        # calculation
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)  # (batch, 2*k_len, dim)
            v = torch.cat((past[1], v), dim=-2)  # (batch, 2*k_len, dim)

        # calculating heads
        q = q.view(
            q.shape[:-1] + (self.num_heads, q.shape[-1] // self.num_heads))  # (batch, q_len, num_heads, dim//num_heads)
        k = k.view(
            k.shape[:-1] + (self.num_heads, k.shape[-1] // self.num_heads))  # (batch, k_len, num_heads, dim//num_heads)
        v = v.view(
            v.shape[:-1] + (self.num_heads, v.shape[-1] // self.num_heads))  # (batch, v_len, num_heads, dim//num_heads)

        # transpose so that the num_head dimension get conserved while calculating attention
        q = q.transpose(-3, -2)  # (batch, num_heads, q_len, dim//num_heads)
        k = k.transpose(-3, -2)  # (batch, num_heads, k_len, dim//num_heads)
        v = v.transpose(-3, -2)  # (batch, num_heads, v_len, dim//num_heads)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.shape[-1])  # (batch, num_heads, q_len, k_len)
        if mask is not None:
            mask = mask.unsqueeze(-3)
            attn_scores += mask.type_as(attn_scores) * attn_scores.new_tensor(-1e5)
        attn_scores = F.softmax(attn_scores, dim=-1)

        # (batch, num_heads, q_len, k_len) @ (batch, num_heads, v_len, dim//num_heads)
        # note: k_len == v_len
        attn_values = torch.matmul(attn_scores, v)  # (batch, num_heads, q_len, dim//num_heads)

        # reshaping it back to the original dimensions
        attn_values = attn_values.transpose(-2, -3)  # (batch, q_len, num_heads, dim//num_heads)
        attn_values = attn_values.contiguous().view(  # (batch, q_len, dim)
            attn_values.shape[:-2] + (self.num_heads * attn_values.shape[-1],))

        attn_values = self.linear(attn_values)
        return attn_values, (k, v)


if __name__ == "__main__":
    q_ = torch.randn(32, 3, 8)
    k_ = torch.randn(32, 3, 8)
    v_ = torch.randn(32, 3, 8)

    past_ = (k_.clone(), v_.clone())
    attn = MultiHeadAttention(8, 8)
    op = attn(q_, k_, v_, past=past_)
    print(op.shape)
