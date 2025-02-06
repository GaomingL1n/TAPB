import pandas as pd
import numpy as np
from sklearn.utils import shuffle

df = pd.read_csv('../datasets/bindingdb/random/train_with_id.csv')

# select pos samples
positive_samples = df[df['Y'] == 1]
negative_samples = df[df['Y'] == 1]

unique_positive_samples_smiles = positive_samples.drop_duplicates(subset='SMILES')
unique_positive_samples_protein = unique_positive_samples_smiles.drop_duplicates(subset='Protein')
collection_A = unique_positive_samples_protein.reset_index(drop=True)


# smiles not in A
all_smiles_in_collection_A = set(collection_A['SMILES'])
negative_candidates = df[~df['SMILES'].isin(all_smiles_in_collection_A)]
# assign a new SMILES as a negative sample to each protein in set A, forming negative sample set B
np.random.seed(0)  
negative_samples_list = []

for protein in collection_A['Protein']:
    candidate_smiles = negative_candidates[negative_candidates['Protein'] != protein]['SMILES']
    if not candidate_smiles.empty:
        selected_smile = candidate_smiles.sample(n=1).iloc[0]
        pr_id = collection_A[collection_A['Protein'] == protein]['pr_id'].values[0]
        negative_samples_list.append({'SMILES': selected_smile, 'Protein': protein, 'Y': 0, 'pr_id': pr_id})

collection_B = pd.DataFrame(negative_samples_list)

# concat
combined_samples = pd.concat([collection_A, collection_B], ignore_index=True)
combined_samples.to_csv('../datasets/bindingdb/random/drug_high_train.csv', index=False)

