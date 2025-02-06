import pandas as pd
from sklearn.utils import shuffle

# 1. 读取CSV文件
df = pd.read_csv('../datasets/bindingdb/random/train_with_id.csv')

# 2. 筛选正样本并去重
positive_samples = df[df['Y'] == 1]
unique_positive_samples_smiles = positive_samples.drop_duplicates(subset='SMILES')
unique_positive_samples_protein = unique_positive_samples_smiles.drop_duplicates(subset='Protein')

# 这里假设我们要先按SMILES去重再按Protein去重
# 如果需要反过来，可以调整顺序
final_unique_positive_samples = unique_positive_samples_protein.reset_index(drop=True)

# 3. 生成负样本
shuffled_smiles = shuffle(final_unique_positive_samples['SMILES'], random_state=42).reset_index(drop=True)
negative_samples = final_unique_positive_samples.copy()
negative_samples['SMILES'] = shuffled_smiles

# 将标签改为0表示负样本
negative_samples['Y'] = 0

# 4. 合并正负样本并保存到新的CSV文件
combined_samples = pd.concat([final_unique_positive_samples, negative_samples], ignore_index=True)
combined_samples.to_csv('../datasets/bindingdb/random/0.5tendedncy_train.csv', index=False)

