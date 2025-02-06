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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
ems2_model_path = '../models/protein/esm2_model'
MODEL_CONFIG_PATH = '../configs/model_config.yaml'
model_configs = dict(load_config_file(MODEL_CONFIG_PATH))
# set_seed(seed=2048)
def randomize_smile(sml):
    """Function that randomizes a SMILES sequence. This was adapted from the
    implemetation of E. Bjerrum 2017, SMILES Enumeration as Data Augmentation
    for Neural Network Modeling of Molecules.
    Args:
        sml: SMILES sequence to randomize.
    Return:
        randomized SMILES sequence or
        nan if SMILES is not interpretable.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        smiles = Chem.MolToSmiles(nm, canonical=False)
        print('random')
        return smiles

    except:
        return sml
# Drug = 'CCCCCCCC\C=C\CCCCCCCC(N)=O'
# Protein = 'MKTLLLLAVIMIFGLLQAHGNLVNFHRMIKLTTGKEAALSYGFYGCHCGVGGRGSPKDATDRCCVTHDCCYKRLEKRGCGTKFLSYKFSNSGSRITCAKQDSCRSQLCECDKAAATCFARNKTTYNKKYQYYSNKHCRGSTPRC'
# Drug = 'CN[C@@H](C)C(=O)N[C@@H]1C(=O)N(Cc2c(C3CC3)cnc3ccccc23)c2ccccc2N(C(C)=O)[C@H]1C'#47
# Protein = 'MRHHHHHHRDHFALDRPSETHADYLLRTGQVVDISDTIYPRNPAMYSEEARLKSFQNWPDYAHLTPRELASAGLYYTGIGDQVQCFACGGKLKNWEPGDFPNCFFVLGRAWSEHRRHRNLNIRSE'
# Drug='O=C(N[C@H]1CCC[C@@H]1OCc1ccccc1)C1CCN(c2nc3cc(Cl)ccc3o2)CC1'
# Protein ='MPNYKLTYFNMRGRAEIIRYIFAYLDIQYEDHRIEQADWPEIKSTLPFGKIPILEVDGLTLHQSLAIARYLTKNTDLAGNTEMEQCHVDAIVDTLDDFMSCFPWAEKKQDVKEQMFNELLTYNAPHLMQDLDTYLGGREWLIGNSVTWADFYWEICSTTLLVFKPDLLDNHPRLVTLRKKVQAIPAVANWIKRRPQTKL'
# Drug='OC(=O)\C=C\C1=CC=C(C=C1)C(O)=O'
# Protein ='MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR'
#pair5
# Drug = 'CCCCC1=C(NC2=NC=CN=C12)C1=CC=C(O)C=C1'
# Protein = 'MQKYEKLEKIGEGTYGTVFKAKNRETHEIVALKRVRLDDDDEGVPSSALREICLLKELKHKNIVRLHDVLHSDKKLTLVFEFCDQDLKKYFDSCNGDLDPEIVKSFLFQLLKGLGFCHSRNVLHRDLKPQNLLINRNGELKLADFGLARAFGIPVRCYSAEVVTLWYRPPDVLFGAKLYSTSIDMWSAGCIFAELANAGRPLFPGNDVDDQLKRIFRLLGTPTEEQWPSMTKLPDYKPYPMYPATTSLVNVVPKLNATGRDLLQNLLKCNPVQRISAEEALQHPYFSDFCPP'
Drug= 'CCCCCCCC\C=C\CCCCCCCC(N)=O'
Protein = 'MKTLLLLAVIMIFGLLQAHGNLVNFHRMIKLTTGKEAALSYGFYGCHCGVGGRGSPKDATDRCCVTHDCCYKRLEKRGCGTKFLSYKFSNSGSRITCAKQDSCRSQLCECDKAAATCFARNKTTYNKKYQYYSNKHCRGSTPRC'
dataset = 'biosnap'
split = 'random'
res = 'test'
stage = 2
model_path = f"../results/{dataset}/{split}/{res}/stage_{stage}_best_epoch_49.pth"
head = model_configs['DrugEncoder']['n_head']
begin = 0
end = 41
# begin = 1
# end = 60


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

attn_map = F.softmax(output['attn_map'][0], dim=-1).squeeze().to('cpu')
# attn_map = output['attn_map'][0].squeeze().to('cpu')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


def sparse_protein_list(begin, end):
    # 创建原始蛋白质编号列表
    Protein = list(range(begin, end))
    # 创建一个与Protein长度相同的空列表，初始值全部为空字符串
    sparse_Protein = [' ' for _ in range(len(Protein))]

    # 遍历Protein列表，每隔四个元素（即每五个元素）将值赋给sparse_Protein
    for i in range(len(Protein)):
        if Protein[i] % 5 == 0:  # 因为我们从0开始计数，所以是i+1
            sparse_Protein[i] = Protein[i]

    return sparse_Protein

# 计算所有 attention_map 的最大值和最小值
# all_attention_maps = torch.stack([attn_map[i].unsqueeze(0)[:, begin:end] for i in range(head)])
vmin = 100
vmax = 0

for i in range(head):
    attention_map = attn_map[i].mean(0).unsqueeze(0)[:, 0:41]
    M = attention_map.max()
    Min = attention_map.min()
    if M > vmax:
        vmax = M
    if vmin > Min:
        vmin = Min
print(vmin)
print('---')
print(vmax)
colors = ['#ffffff', '#db9ea3']
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=8)
for i in range(head):
    attention_map = attn_map[i].mean(0).unsqueeze(0)[:,begin:end]
    fig, ax = plt.subplots(figsize=(15, 0.6))
    cax = ax.imshow(attention_map, cmap=cmap, aspect=1, vmin=vmin, vmax=vmax)
    ax.set_yticks([])
    if i == 7:
        Protein = sparse_protein_list(begin+21, end+21)
        ax.set_xticks(list(range(end - begin)))  # 设置x轴刻度位置
        ax.set_xticklabels(Protein)
    else:
        ax.set_xticks([])
    # 调整子图间距，确保标签完全显示
    plt.subplots_adjust(bottom=0.4)

    plt.savefig(f'./attn_map_pair5_head{i}-0-20.png')
    plt.close(fig)



