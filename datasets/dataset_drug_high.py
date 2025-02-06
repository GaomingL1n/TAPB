import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# 1. 读取CSV文件
df = pd.read_csv('../datasets/bindingdb/random/train_with_id.csv')

# 2. 筛选正样本并去重
positive_samples = df[df['Y'] == 1]
negative_samples = df[df['Y'] == 1]

unique_positive_samples_smiles = positive_samples.drop_duplicates(subset='SMILES')
unique_positive_samples_protein = unique_positive_samples_smiles.drop_duplicates(subset='Protein')
collection_A = unique_positive_samples_protein.reset_index(drop=True)


# 3. 取出不在集合A中出现过的SMILES
all_smiles_in_collection_A = set(collection_A['SMILES'])
negative_candidates = df[~df['SMILES'].isin(all_smiles_in_collection_A)]
# 4. 为集合A中的每个蛋白质分配一个新的SMILES作为负样本，构成负样本集合B
np.random.seed(0)  # 设置随机种子以确保结果可重复
negative_samples_list = []

for protein in collection_A['Protein']:
    candidate_smiles = negative_candidates[negative_candidates['Protein'] != protein]['SMILES']
    if not candidate_smiles.empty:
        selected_smile = candidate_smiles.sample(n=1).iloc[0]
        pr_id = collection_A[collection_A['Protein'] == protein]['pr_id'].values[0]
        negative_samples_list.append({'SMILES': selected_smile, 'Protein': protein, 'Y': 0, 'pr_id': pr_id})

collection_B = pd.DataFrame(negative_samples_list)

# 4. 合并正负样本并保存到新的CSV文件
combined_samples = pd.concat([collection_A, collection_B], ignore_index=True)
combined_samples.to_csv('../datasets/bindingdb/random/drug_high_train.csv', index=False)

