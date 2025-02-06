import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('../datasets/bindingdb/random/train.csv')

# 定义计算倾向性的函数
def calculate_tendency(df, column):
    # 计算指定列的倾向性
    tendencies = df.groupby(column)['Y'].mean().round(1)
    return tendencies

# 计算SMILES和Protein的倾向性
smiles_tendencies = calculate_tendency(df, 'SMILES')
protein_tendencies = calculate_tendency(df, 'Protein')

# 统计每个四舍五入后的倾向性出现的次数
smiles_frequency = smiles_tendencies.value_counts(normalize=True).sort_index()
protein_frequency = protein_tendencies.value_counts(normalize=True).sort_index()

# 将比例转换为百分比形式
# smiles_percentages = smiles_frequency * 100
# protein_percentages = protein_frequency * 100

# 将Pandas Series转换为NumPy数组
smiles_x = smiles_frequency.index.to_numpy()
smiles_y = smiles_frequency.values
protein_x = protein_frequency.index.to_numpy()
protein_y = protein_frequency.values
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

# 创建图表
plt.figure(figsize=(10, 6))
#9bd7f3#f2a1a7
# 绘制Protein倾向性折线图
plt.plot(protein_x, protein_y, color='#9B123C', marker='o', label='Target')
plt.fill_between(protein_x,protein_y, color='#F6A6BF', alpha=0.6)
# 绘制SMILES倾向性折线图
plt.plot(smiles_x, smiles_y, color='#182F8C', marker='o', label='SMILES')
plt.fill_between(smiles_x,smiles_y,color='#B9D2F1', alpha=0.6,)

# 设置图表标题和轴标签
plt.xlabel('Tendency  $t_i$')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(0,1)
plt.ylim(0, max(smiles_y.max(),protein_y.max())+0.1)
# 显示网格
# plt.grid(True, linestyle='--', color='gray')
plt.savefig('../visualization/bindingdbrandom.svg',format='svg')
# 显示图表
plt.show()

