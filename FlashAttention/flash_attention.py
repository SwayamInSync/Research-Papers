"""
Note: It's an efficient way of calculating the softmax and attention weights,
so perform all the necessary actions before.

This implementation assumes, you already had the q,k and v vectors calculated
"""

import torch


def flash_attn(q, k, v, blocksizes, dim):
    """
    :param q: query, it should be after getting multiplied by proj_q
    :param k: key, it should be after getting multiplied by proj_k
    :param v: value, it should be after getting multiplied by proj_v
    :param blocksizes: block sizes (Br, Bc) as mentioned in paper
    :param dim: dimension of q,k,v
    """
    q_len, _ = q.shape
    k_len, _ = k.shape
    Br, Bc = blocksizes

    Tr = q_len // Br + q_len % Br
    Tc = k_len // Bc + k_len % Bc

    q = torch.chunk(q, Tr, dim=0)  # (Br, d) x Tr
    k = torch.chunk(k, Tc, dim=0)  # (Bc, d) x Tc
    v = torch.chunk(v, Tc, dim=0)  # (Bc, d) x Tc

    O = list(torch.chunk(torch.zeros(q_len, dim), Tr, dim=0))  # (Br, seq_len, d) x Tr
    l = torch.zeros(q_len)
    m = torch.ones(q_len) * -1 * torch.inf

    for j in range(Tc):
        K_j, V_j = k[j], v[j]

        for i in range(Tr):
            Q_i = q[i]
            O_i = O[i]
            l_i, m_i = l[i:i + Br], m[i: i + Br]

            S_ij = torch.matmul(Q_i, K_j.transpose(0, 1))  # (Br, Bc)
            m_ij, _ = torch.max(S_ij, dim=1)  # (Br,)
            P_ij = torch.exp(S_ij - m_ij)  # (Br, Bc)
            l_ij, _ = torch.max(P_ij, dim=1)  # (Br,)

            m_i_new = torch.max(m_i, m_ij)
            l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.exp(m_ij - m_i_new) * l_ij

            l_i_new_inv = torch.diag(1 / l_i_new)
            t1 = torch.diag(l_i) * torch.exp(m_i - m_i_new)
            t1 = torch.matmul(t1, O_i)
            t2 = torch.exp(m_ij - m_i_new).unsqueeze(1) * torch.matmul(P_ij, V_j)

            O[i] = torch.matmul(l_i_new_inv, t1 + t2)
            l[i:i + Br] = l_i_new
            m[i: i + Br] = m_i_new
    return torch.cat(O, dim=0)


if __name__ == "__main__":
    q = torch.randn(10, 64)
    k = torch.randn(10, 64)
    v = torch.randn(10, 64)
    blocksizes = (2, 2)
    op = flash_attn(q, k, v, blocksizes, 64)
    print(op.shape)
