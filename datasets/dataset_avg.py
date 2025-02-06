import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv('../datasets/bindingdb/random/train_with_id.csv')

# select pos samples
positive_samples = df[df['Y'] == 1]
unique_positive_samples_smiles = positive_samples.drop_duplicates(subset='SMILES')
unique_positive_samples_protein = unique_positive_samples_smiles.drop_duplicates(subset='Protein')

final_unique_positive_samples = unique_positive_samples_protein.reset_index(drop=True)

# generate neg pairs
shuffled_smiles = shuffle(final_unique_positive_samples['SMILES'], random_state=42).reset_index(drop=True)
negative_samples = final_unique_positive_samples.copy()
negative_samples['SMILES'] = shuffled_smiles

# labels = 0
negative_samples['Y'] = 0

# concat
combined_samples = pd.concat([final_unique_positive_samples, negative_samples], ignore_index=True)
combined_samples.to_csv('../datasets/bindingdb/random/0.5tendedncy_train.csv', index=False)

