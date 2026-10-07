import torch
import torch.nn.functional as F


def sdpa_dense(q, k, v):
    """q, k, v: [B, N, H, C] -> [B, N, H, C]"""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2)


def sdpa_varlen(q, k, v, q_seqlens, kv_seqlens):
    """q: [Mq, H, C], k/v: [Mkv, H, C], concatenated variable-length sequences -> [Mq, H, C]"""
    outs = []
    qs, ks = 0, 0
    for lq, lkv in zip(q_seqlens, kv_seqlens):
        qi = q[qs:qs + lq].transpose(0, 1).unsqueeze(0)
        ki = k[ks:ks + lkv].transpose(0, 1).unsqueeze(0)
        vi = v[ks:ks + lkv].transpose(0, 1).unsqueeze(0)
        outs.append(F.scaled_dot_product_attention(qi, ki, vi)[0].transpose(0, 1))
        qs += lq
        ks += lkv
    return torch.cat(outs, dim=0)
