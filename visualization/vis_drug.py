import matplotlib.pyplot as plt
import torch
import pickle
from rdkit import Chem
import numpy as np
from models.transformer_dti import TransformerDTI
from utils.utils import set_seed, load_config_file
from transformers import AutoTokenizer, EsmModel
import torch.nn.functional as F
import matplotlib.colors as mcolors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
ems2_model_path = '../models/protein/esm2_model'
MODEL_CONFIG_PATH = '../configs/model_config.yaml'
model_configs = dict(load_config_file(MODEL_CONFIG_PATH))
# set_seed(seed=2048)
# Protein = 'MAKEWGYASHNGPDHWHELFPNAKGENQSPVELHTKDIRHDPSLQPWSVSYDGGSAKTILNNGKTCRVVFDDTYDRSMLRGGPLPGPYRLRQFHLHWGSSDDHGSEHTVDGVKYAAELHLVHWNPKYNTFKEALKQRDGIAVIGIFLKIGHENGEFQIFLDALDKIKTKGKEAPFTKFDPSCLFPACRDYWTYQGSFTTPPCEECIVWLLLKEPMTVSSDQMAKLRSLLSSAENEPPVPLVSNWRPPQPINNRVVRASFK'
Drug = 'CCCCCCCC\C=C\CCCCCCCC(N)=O'
Protein = 'MKTLLLLAVIMIFGLLQAHGNLVNFHRMIKLTTGKEAALSYGFYGCHCGVGGRGSPKDATDRCCVTHDCCYKRLEKRGCGTKFLSYKFSNSGSRITCAKQDSCRSQLCECDKAAATCFARNKTTYNKKYQYYSNKHCRGSTPRC'


dataset = 'biosnap'
split = 'random'
res = 'test'
stage = 2
model_path = f"../results/{dataset}/{split}/{res}/stage_{stage}_best_epoch_49.pth"
head = model_configs['DrugEncoder']['n_head']
topk = 5
colors = ['#ffffff', '#db9ea3']

checkpoint = torch.load(model_path)

if stage == 1:
    model = TransformerDTI(model_configs=model_configs).to(device)
else:
    pr_confounder_path = f"../results/{dataset}/{split}/{res}/pr_confoudner.pkl"
    confounder_path = open(pr_confounder_path, 'rb')
    confounder = pickle.load(confounder_path)
    pr_confounder = torch.from_numpy(confounder['cluster_centers']).to(device)
    model = TransformerDTI(
        pr_confounder=pr_confounder,
        model_configs=model_configs).to(device)

model.load_state_dict(checkpoint,strict=False)
model = model.to(device)
model.eval()

mol_path = '../models/drug/molformer'
drug_tokenizer = AutoTokenizer.from_pretrained(mol_path, trust_remote_code=True)
pr_tokenizer = AutoTokenizer.from_pretrained(ems2_model_path)
input_drugs = drug_tokenizer(Drug, return_tensors="pt").to(device)
pr_input = pr_tokenizer(Protein, return_tensors="pt").to(device)
esm = EsmModel.from_pretrained(ems2_model_path).to(device)

with torch.no_grad():
    outputs = esm(**pr_input)
    input_proteins = outputs.last_hidden_state.to(device)
    pr_mask = pr_input['attention_mask'].to(device)
    output = model(input_drugs, input_proteins, pr_mask=pr_mask)

attn_map_drug = F.softmax(output['attn_map'], dim=-1).squeeze().to('cpu')
plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 12

from rdkit.Chem import AllChem, Draw
text = ['C','N','O','S']
mol = Chem.MolFromSmiles(Drug)
drug_ids = input_drugs['input_ids'].squeeze().tolist()
drug_tokens = drug_tokenizer.convert_ids_to_tokens(drug_ids)
mapping = list()
atom=0

for i in range(len(drug_tokens)):
    if drug_tokens[i] in text:
        mapping.append(atom)
        atom+=1
    else:
        attn_map_drug[:,i,:]=0
        mapping.append(-1)

for i in range(head):
    attn_map = attn_map_drug[i].mean(1)
    values, indices = torch.topk(attn_map, topk)
    atom_indices = [mapping[index] for index in indices.tolist()]
    AllChem.Compute2DCoords(mol)

    img = Draw.MolToImage(mol, highlightAtoms=atom_indices, size=(800, 800))
    img.save(f'./pair5_{i}.png')