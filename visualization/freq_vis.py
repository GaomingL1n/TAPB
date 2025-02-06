import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../datasets/bindingdb/random/train.csv')

def calculate_tendency(df, column):
    tendencies = df.groupby(column)['Y'].mean().round(1)
    return tendencies

smiles_tendencies = calculate_tendency(df, 'SMILES')
protein_tendencies = calculate_tendency(df, 'Protein')

smiles_frequency = smiles_tendencies.value_counts(normalize=True).sort_index()
protein_frequency = protein_tendencies.value_counts(normalize=True).sort_index()

# smiles_percentages = smiles_frequency * 100
# protein_percentages = protein_frequency * 100

smiles_x = smiles_frequency.index.to_numpy()
smiles_y = smiles_frequency.values
protein_x = protein_frequency.index.to_numpy()
protein_y = protein_frequency.values
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

plt.figure(figsize=(10, 6))
plt.plot(protein_x, protein_y, color='#9B123C', marker='o', label='Target')
plt.fill_between(protein_x,protein_y, color='#F6A6BF', alpha=0.6)
plt.plot(smiles_x, smiles_y, color='#182F8C', marker='o', label='SMILES')
plt.fill_between(smiles_x,smiles_y,color='#B9D2F1', alpha=0.6,)

plt.xlabel('Tendency  $t_i$')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(0,1)
plt.ylim(0, max(smiles_y.max(),protein_y.max())+0.1)
# plt.grid(True, linestyle='--', color='gray')
plt.savefig('../visualization/xxxx.svg',format='svg')
plt.show()

