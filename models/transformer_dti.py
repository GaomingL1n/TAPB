import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Transformer import TransformerEncoder, TransformerDecoder
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

class TransformerDTI(nn.Module):
    def __init__(self, model_configs, pr_confounder=None, drug_confounder=None, fusion_confounder=None):
        super().__init__()
        n_embd = model_configs['DrugEncoder']['n_embd']
        n_head = model_configs['DrugEncoder']['n_head']
        self.n_head = n_head
        vocab_size = model_configs['DrugEncoder']['vocab_size']
        self.precompute_freqs_cis = precompute_freqs_cis(n_embd // n_head, 4000)
        self.drug_encoder = TransformerEncoder(config=model_configs['DrugEncoder'])
        self.pr_linear = nn.Linear(1280, n_embd)

        self.decoder = TransformerDecoder(config=model_configs['TransformerDeocder'])

        self.tem = 1.0
        # self.weight = None

        self.MLMHead = MLMHead(n_embd, vocab_size)
        self.MLP2 = nn.Sequential(
            nn.Linear(n_embd*2, n_embd),
            nn.ReLU(),
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, 1)
        )

        self.pr_confounder = pr_confounder
        if self.pr_confounder is not None:
            self.pr_confounder = pr_confounder.float().permute(1,0)
            self.confounder_W_q = nn.Linear(1280, 1280)
            self.confounder_W_k = nn.Linear(1280, 1280)

        self.drug_confounder = drug_confounder
        if self.drug_confounder is not None:
            self.drug_confounder = drug_confounder.float().permute(1, 0)
            self.confounder_W_q2 = nn.Linear(n_embd, n_embd)
            self.confounder_W_k2 = nn.Linear(n_embd, n_embd)

        self.fusion_confounder = fusion_confounder
        if self.fusion_confounder is not None:
            self.fusion_confounder = fusion_confounder.float().permute(1, 0)
            self.confounder_W_q3 = nn.Linear(n_embd, n_embd)
            self.confounder_W_k3 = nn.Linear(n_embd, n_embd)
            freeze(self.drug_encoder)
            freeze(self.pr_linear)
            freeze(self.confounder_W_q)
            freeze(self.confounder_W_k)
            freeze(self.confounder_W_q2)
            freeze(self.confounder_W_k2)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     if self.tem is None:
    #         self.tem = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
    #         self.tem.data.fill_(20.0)
    #         self.weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
    #         self.weight.data.fill_(0.01)

    def encode_protein(self, pr_f, protein_padding_mask):
        if self.pr_confounder is not None:
            pr_f = self.pr_backdoor_adjust(pr_f, protein_padding_mask)
        pr_f = self.pr_linear(pr_f)
        return pr_f

    def encode_drug(self, drug_id, drug_padding_mask, freqs_cis):
        bz, len_d = drug_id.size()
        drug_attn_mask = ~drug_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.n_head, len_d, -1)
        drug_f = self.drug_encoder(drug_id, drug_attn_mask, freqs_cis)
        if self.drug_confounder is not None:
            drug_f = self.drug_backdoor_adjust(drug_f, drug_padding_mask)
        return drug_f

    def decode(self, drug_f, pr_f, drug_padding_mask, protein_padding_mask):
        bz, len_d, _ = drug_f.size()
        drug_attn_mask = ~drug_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.n_head, len_d, -1)
        cross_attn_mask = ~protein_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.n_head, len_d, -1)
        fusion_f, attention_map = self.decoder(src=pr_f, tgt=drug_f, self_attn_mask=drug_attn_mask,
                                               cross_attn_mask=cross_attn_mask)
        return fusion_f, attention_map

    def pr_backdoor_adjust(self, pr_f, mask):
        device = pr_f.device
        bz, _, _ = pr_f.size()
        len_c, _ = self.pr_confounder.size()
        mask = mask.bool().unsqueeze(-1).expand(-1, -1, len_c)
        Q = self.confounder_W_q(pr_f)
        K = self.confounder_W_k(self.pr_confounder.unsqueeze(0).expand(bz,-1,-1))
        attention = torch.matmul(Q, K.permute(0, 2, 1)) / torch.sqrt(
            torch.tensor(K.shape[1], dtype=torch.float32, device=device))
        attention = attention.masked_fill(mask, -1e10)
        attention = F.softmax(attention, dim=-1)
        Q = torch.matmul(attention, K)
        pr_f = torch.cat((pr_f, Q),dim=-1)
        return pr_f

    def drug_backdoor_adjust(self, drug_f, mask):
        device = drug_f.device
        bz, _, _ = drug_f.size()
        len_c, _ = self.drug_confounder.size()
        mask = mask.bool().unsqueeze(-1).expand(-1, -1, len_c)
        Q = self.confounder_W_q2(drug_f)
        K = self.confounder_W_k2(self.drug_confounder.unsqueeze(0).expand(bz,-1,-1))
        attention = torch.matmul(Q, K.permute(0, 2, 1)) / torch.sqrt(torch.tensor(K.shape[2], dtype=torch.float32, device=device))
        attention = attention.masked_fill(mask, -1e10)
        attention = F.softmax(attention, dim=-1)
        Q = torch.matmul(attention, K)
        drug_f = torch.cat((drug_f, Q), dim=-1)
        return drug_f

    def multi_encoder_backdoor_adjust(self, cls):
        device = cls.device
        Q = self.confounder_W_q3(cls)
        K = self.confounder_W_k3(self.fusion_confounder)
        attention = torch.matmul(Q, K.permute(1, 0)) / torch.sqrt(
            torch.tensor(K.shape[1], dtype=torch.float32, device=device))
        attention = F.softmax(attention, dim=-1)
        Q = torch.matmul(attention, K)
        cls = cls + Q
        return cls

    def forward(self, drug_id, drug_padding_mask, pr_f, protein_padding_mask, itm=False, mlm=False):
            bz, len_d = drug_id.size()
            drug_f = self.encode_drug(drug_id, drug_padding_mask, self.precompute_freqs_cis[:len_d, :].to(drug_id.device))

            if mlm :
                drug_logits = self.MLMHead(drug_f).permute(0, 2, 1)
                return {'drug_logits': drug_logits}
            else:
                pr_f = self.encode_protein(pr_f, protein_padding_mask)
            if itm:
                drug_fs = drug_f[:, 0, :]
                pr_fs = pr_f[:, 0, :]
                drug_fs = F.normalize(drug_fs, dim=1)
                pr_fs = F.normalize(pr_fs, dim=1)
                sim = drug_fs @ pr_fs.T / self.tem
                with torch.no_grad():
                    weights = F.softmax(sim, dim=1)

                    weights.fill_diagonal_(0)
                pr_f_neg, pr_mask, drug_f_neg, drug_mask = [],[],[],[]
                # for b in range(bz):
                #     neg_idx = torch.multinomial(weights[b], 1).item()
                #     pr_f_neg.append(pr_f[neg_idx])
                #     pr_mask.append(protein_padding_mask[neg_idx])
                #
                #     weights = weights.T
                #     neg_idx = torch.multinomial(weights[b], 1).item()
                #     drug_f_neg.append(drug_f[neg_idx])
                #     drug_mask.append(drug_padding_mask[neg_idx])
                #
                # pr_neg = torch.stack(pr_f_neg, dim=0)
                # drug_neg = torch.stack(drug_f_neg, dim=0)
                # pr_mask = torch.stack(pr_mask, dim=0)
                # drug_mask = torch.stack(drug_mask, dim=0)
                neg_idx_pr = torch.multinomial(weights, 1).flatten()  # 结果形状是 (bz,)
                neg_idx_drug = torch.multinomial(weights.T, 1).flatten()  # 结果形状是 (bz,)

                # 使用索引提取负样本和掩码
                pr_neg = pr_f[neg_idx_pr]
                drug_neg = drug_f[neg_idx_drug]

                pr_mask = protein_padding_mask[neg_idx_pr]
                drug_mask = drug_padding_mask[neg_idx_drug]

                drug_f = torch.cat((drug_f, drug_f, drug_neg), dim=0)
                pr_f = torch.cat((pr_f, pr_neg, pr_f), dim=0)
                drug_padding_mask = torch.cat((drug_padding_mask, drug_padding_mask, drug_mask),dim=0)
                protein_padding_mask = torch.cat((protein_padding_mask, pr_mask, protein_padding_mask),dim=0)
            fusion_f, attention_map = self.decode(drug_f, pr_f, drug_padding_mask, protein_padding_mask)
            cls = fusion_f[:, 0, :]
            logits = self.MLP2(cls)
            return {'logits': logits, 'attention_map': attention_map}
