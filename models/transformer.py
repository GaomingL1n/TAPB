import torch
import torch.nn as nn
import math
from models.attention import MultiHeadSelfAttention, SDPA
import torch.nn.functional as F
from utils.utils import inverse_sigmoid

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_attention_fn(attention, d_model, n_heads, dropout,cro=False):
    """Return an activation function given a string"""
    if attention == 'SDPA':
        return SDPA(d_model=d_model, n_heads=n_heads, dropout=dropout)
    if attention == 'OriginMHA':
        return MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
    raise RuntimeError(F"attention should be MHA/DeformableMHA, not {attention}.")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4000):
        """Positional Encoding.
        Args:
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class PositionWiseFeedforward(nn.Module):
    def __init__(self, d_model, pf_dim, dropout, activation):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Conv1d(d_model, pf_dim, 1),
            _get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Conv1d(pf_dim, d_model, 1)
        )

    def forward(self, x):
        # x = [batch size, sent len, hid dim]
        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]
        x = self.FFN(x)
        # x = [batch size, pf dim, sent len]
        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=64, pf_dim=64, dropout=0.1,
                 n_heads=8, activation='gelu', attention='MHA'):
        super().__init__()
        self.attention = attention
        self.self_attn = _get_attention_fn(attention, d_model, n_heads, dropout)
        self.FFN = PositionWiseFeedforward(d_model, pf_dim, dropout, activation)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask=None,freqs_cis=None):

        # self attention
        if self.attention == 'MHA':
            src = src + self.dropout1(
                self.self_attn(query=src, key=src, value=src, key_padding_mask=src_key_padding_mask)[0])
            #[len,bz,dim]
        else:
            src = src + self.dropout1(
                self.self_attn(Q=src, K=src, V=src,  mask=src_key_padding_mask, freqs_cis=freqs_cis))
        src = self.ln1(src)

        # ffn
        src = self.ln2(src + self.dropout2(self.FFN(src)))

        return src




class TransformerEncoder(nn.Module):

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        # self.positional_Embedding = PositionalEncoding(d_model=config['n_embd'])
        self.Embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model=config['d_model'],
                                     activation=config['activation'],
                                     pf_dim=config['d_model'],
                                     n_heads=config['n_head'],
                                     dropout=config['dropout'],
                                     attention=config['attention'])
             for _ in range(config['n_layer'])])


    def forward(self, drug_f, padding_mask, freqs_cis):
        drug_f = self.Embeddings(drug_f)
        # drug_f = torch.randn_like(drug_f)
        for layer in self.layers:
            drug_f = layer(src=drug_f, src_key_padding_mask=padding_mask, freqs_cis=freqs_cis)
        return drug_f

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=64, pf_dim=64, dropout=0.1,
                 n_heads=8, activation='gelu', attention='MHA'):
        super().__init__()

        self.n_heads = n_heads
        self.self_attn = _get_attention_fn(attention, d_model, n_heads, dropout, cro=False)
        self.cross_attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.FFN = PositionWiseFeedforward(d_model, pf_dim, dropout, activation)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self, tgt, src, self_attn_mask, cross_attn_mask, freqs_cis=None,cf=False):
        """
        :param tgt: [batch_size, compound len, atom_dim]
        :param src: [batch_size, protein len, hid_dim] # encoder output
        :param tgt_mask: [batch size, compound sent len]
        :param src_mask: [batch size, protein len]

        :return tg: [batch_size, compound len, atom_dim]
        """
        tgt = self.ln1(tgt + self.dropout1(self.self_attn(tgt, tgt, tgt, self_attn_mask, freqs_cis)))
        tgt_cross, attention_map = self.cross_attn(tgt, src, src, cross_attn_mask, self_attn=False)
        tgt = self.ln2(tgt + self.dropout2(tgt_cross))
        tgt = self.ln3(tgt + self.dropout3(self.FFN(tgt)))

        return tgt, attention_map

class TransformerDecoder(nn.Module):
    """ ."""

    def __init__(self, config):
        super().__init__()
        # self.objq = nn.Parameter(torch.randn(1,64))
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=config['d_model'],
                                     activation=config['activation'],
                                     pf_dim=config['d_model'],
                                     n_heads=config['n_head'],
                                     dropout=config['dropout'],
                                     attention=config['attention'])
             for _ in range(config['n_layer'])])

    def forward(self, src, tgt, self_attn_mask=None, cross_attn_mask=None, freqs_cis=None,cf=False):
        # tgt = drug
        # src = protein
        # attn = list()
        # bz, len_d,_ = tgt.size()
        # objq = self.objq.unsqueeze(0).expand(bz,len_d,-1)
        for layer in self.layers:
            tgt, attention_map = layer(tgt, src, self_attn_mask, cross_attn_mask, freqs_cis, cf)
            # attn.append(attention_map)
        return tgt, attention_map