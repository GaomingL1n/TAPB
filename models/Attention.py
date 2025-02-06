import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, xavier_uniform_, constant_

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
class SDPA(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

        self.do = dropout

        self.scale = math.sqrt(d_model // n_heads)

    def forward(self, Q, K, V, mask, freqs_cis=None):
        bz, len_q, _ = Q.size()

        Q = self.w_q(Q).view(bz, -1, self.n_heads, self.d_model // self.n_heads)
        K = self.w_k(K).view(bz, -1, self.n_heads, self.d_model // self.n_heads)
        V = self.w_v(V).view(bz, -1, self.n_heads, self.d_model // self.n_heads)
        if freqs_cis is not None:
            Q,K = apply_rotary_emb(Q, K, freqs_cis)
        # Q, K, V = [batch size, sent len, hid dim]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            Q = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, scale=self.scale, dropout_p=self.do)
        Q = Q.permute(0, 2, 1, 3).contiguous().view(bz, -1, self.n_heads * (self.d_model // self.n_heads))
        Q = self.fc(Q)
        # x = [batch size, sent len_Q, hid dim]

        return Q


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([d_model // n_heads]))

    def forward(self, Q, K, V, mask, freqs_cis=None, self_attn=True):
        bz, len_q, _ = Q.size()
        dtype, device = Q.dtype, Q.device
        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(Q).view(bz, -1, self.n_heads, self.d_model // self.n_heads)
        K = self.w_k(K).view(bz, -1, self.n_heads, self.d_model // self.n_heads)
        V = self.w_v(V).view(bz, -1, self.n_heads, self.d_model // self.n_heads)

        # Q, K, V = [batch size, sent len, hid dim]
        if freqs_cis is not None:
            Q, K = apply_rotary_emb(Q, K, freqs_cis)
        # Q, K, V = [batch size, sent len, hid dim]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(device)
        #energy = energy + adj.unsqueeze(1).expand(-1,self.n_heads,-1,-1)
        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            #energy = energy.transpose(1, 3).masked_fill(mask[..., None, None], -1e10).transpose(1, 3)
            energy = energy.masked_fill(mask, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        Q = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        Q = Q.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        Q = Q.view(bz, -1, self.n_heads * (self.d_model // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        Q = self.fc(Q)

        # x = [batch size, sent len_Q, hid dim]
        if self_attn:
            return Q
        else:
            return Q, attention
