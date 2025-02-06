import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import TransformerEncoder, TransformerDecoder

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

class MLMHead(nn.Module):
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, n_embd)
        self.activation = nn.GELU()
        self.ln = nn.LayerNorm(n_embd)
        self.linear2 = nn.Linear(n_embd, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.linear2.bias = self.bias

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.ln(x)
        logits = self.linear2(x)
        return logits

# TAPB FRAMEWORK
class TransformerDTI(nn.Module):
    def __init__(self, model_configs, pr_confounder=None):
        super().__init__()
        # Params
        d_model = model_configs['DrugEncoder']['d_model']
        n_head = model_configs['DrugEncoder']['n_head']
        vocab_size = model_configs['DrugEncoder']['vocab_size']
        self.n_head = n_head

        # Drug
        self.precompute_freqs_cis = precompute_freqs_cis(d_model // n_head, 4000)
        self.drug_encoder = TransformerEncoder(config=model_configs['DrugEncoder'])
        self.MLMHead = MLMHead(d_model, vocab_size)

        # Target
        self.pr_linear = nn.Linear(1280, d_model)

        # Fusion
        self.decoder = TransformerDecoder(config=model_configs['TransformerDeocder'])
        self.classifier = nn.Linear(d_model, 2)

        # Stage 2 training
        self.pr_confounder = pr_confounder
        if self.pr_confounder is not None:
            self.pr_confounder = pr_confounder.float().permute(1,0)
            # self.decoder2 = TransformerDecoder(config=model_configs['TransformerDeocder'])
            self.classifier2 = nn.Linear(d_model, 2)
            freeze(self.drug_encoder)
            freeze(self.pr_linear)

    def backdoor_adjust(self, pr_f):
        device = pr_f.device
        Q = pr_f
        K = self.pr_confounder
        A = torch.mm(K, Q.transpose(0, 1))
        A = F.softmax(A / torch.sqrt(torch.tensor(K.shape[1], dtype=torch.float32, device=device)),0)  # normalize attention scores, A in shape N x C,
        Q = torch.mm(A.transpose(0, 1), self.pr_confounder)
        return Q

    def encode_drug(self, input_drugs, freqs_cis):
        drug_id = input_drugs['input_ids']
        drug_padding_mask = ~input_drugs['attention_mask'].bool()
        bz, len_d = drug_id.size()
        drug_attn_mask = drug_padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.n_head, len_d, -1)
        drug_f = self.drug_encoder(drug_id, drug_attn_mask, freqs_cis[:len_d, :].to(drug_id.device))
        return drug_f

    def encode_protein(self, pr_f):
        pr_f = self.pr_linear(pr_f)
        return pr_f

    def decode(self, drug_f, pr_f, drug_padding_mask, protein_padding_mask):
        bz, len_d, _ = drug_f.size()
        drug_attn_mask = ~drug_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.n_head, len_d, -1)
        cross_attn_mask = ~protein_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.n_head, len_d, -1)
        if self.pr_confounder is None:
            fusion_f, attention_map = self.decoder(src=pr_f, tgt=drug_f, self_attn_mask=drug_attn_mask,
                                                   cross_attn_mask=cross_attn_mask)
        else:
            fusion_f, attention_map = self.decoder(src=pr_f, tgt=drug_f, self_attn_mask=drug_attn_mask,
                                                   cross_attn_mask=cross_attn_mask)
        return fusion_f.mean(1), attention_map

    def forward(self, input_drugs, input_proteins, pr_mask=None, masked_drugs=None):
            # encode
            drug_f = self.encode_drug(input_drugs, self.precompute_freqs_cis)
            pr_f = self.encode_protein(input_proteins)
            # decode
            fusion_f, attn_map = self.decode(drug_f, pr_f, input_drugs['attention_mask'], pr_mask)

            if self.pr_confounder is not None:
                fusion_f = self.backdoor_adjust(fusion_f)
                logits = self.classifier2(fusion_f)
            else:
                logits = self.classifier(fusion_f)
            logits = F.softmax(logits, dim=-1)

            # mlm
            drug_mlm_logits = None
            if masked_drugs is not None:
                drug_f_mlm = self.encode_drug(masked_drugs, self.precompute_freqs_cis)
                drug_mlm_logits = self.MLMHead(drug_f_mlm).permute(0, 2, 1)

            return {'logits': logits, 'fusion_f': fusion_f, 'attn_map': attn_map, 'drug_mlm_logits': drug_mlm_logits}
