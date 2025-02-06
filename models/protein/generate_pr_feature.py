import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

print('start')
# for example, dataset = 'biosnap/random/'
dataset = 'your_data_set/split/'
dataset_path = '../../datasets/'+dataset
save_path = dataset_path+'pr_f_1280_2000.pkl'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dfs = [pd.read_csv(f'{dataset_path}{dataset}_with_id.csv') for dataset in ['train', 'val', 'test']]
df = pd.concat(dfs)

ems2_model_path = './esm2_model'
tokenizer = AutoTokenizer.from_pretrained(ems2_model_path)
model = EsmModel.from_pretrained(ems2_model_path).to(device)
model.eval()  # disables dropout for deterministic results
prlist = list()

for protein_id in tqdm(df['pr_id'].unique(), desc='Processing'):
    protein_seq = df[df['pr_id'] == protein_id]['Protein'].iloc[0]
    inputs = tokenizer(protein_seq, return_tensors="pt", truncation = True, max_length=2000).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    sr = outputs.last_hidden_state.squeeze()
    prlist.append(sr)

file = open(save_path, 'wb')
pickle.dump(prlist, file)
print('finish')
