# 导入必要的库
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
# from pymoo.factory import get_sampling, get_crossover, get_mutation
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.font_manager as fm
import matplotlib

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# 定义是否使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用的设备: {device}')

# 1. 数据加载与预处理

# 读取Excel文件中的各个工作表
sheet1 = pd.read_excel('trainingset.xlsx', sheet_name='材料1')
sheet2 = pd.read_excel('trainingset.xlsx', sheet_name='材料2')
sheet3 = pd.read_excel('trainingset.xlsx', sheet_name='材料3')
sheet4 = pd.read_excel('trainingset.xlsx', sheet_name='材料4')

# 为每个工作表添加‘磁芯材料’列
sheet1.insert(3, '磁芯材料', '材料1')
sheet2.insert(3, '磁芯材料', '材料2')
sheet3.insert(3, '磁芯材料', '材料3')
sheet4.insert(3, '磁芯材料', '材料4')

# 合并所有工作表
data = pd.concat([sheet1, sheet2, sheet3, sheet4], ignore_index=True)

# 提取特征和目标变量
numerical_features = ['温度，oC', '频率，Hz']
categorical_features = ['磁芯材料', '励磁波形']
sequence_features = data.columns[5:]  # 假设序列特征从第6列开始

X = data[numerical_features + categorical_features + list(sequence_features)]
y = data['磁芯损耗，w/m3'].values

# 拆分数据集为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 对类别型特征进行One-Hot编码
onehot_encoder = OneHotEncoder(sparse_output=False)
categorical_encoded = onehot_encoder.fit_transform(X_train[categorical_features])
X_val_categorical_encoded = onehot_encoder.transform(X_val[categorical_features])

# 对数值型特征进行Min-Max标准化
scaler_num = MinMaxScaler()
numerical_scaled = scaler_num.fit_transform(X_train[numerical_features])
X_val_numerical_scaled = scaler_num.transform(X_val[numerical_features])

# 对磁通密度序列特征进行Min-Max标准化
scaler_flux = MinMaxScaler()
flux_density = X_train[sequence_features].values
flux_density_scaled = scaler_flux.fit_transform(flux_density)
X_val_flux_density_scaled = scaler_flux.transform(X_val[sequence_features].values)

# 合并所有处理后的特征
X_train_processed = np.hstack([numerical_scaled, categorical_encoded, flux_density_scaled])
X_val_processed = np.hstack([X_val_numerical_scaled, X_val_categorical_encoded, X_val_flux_density_scaled])

# 对目标变量进行标准化
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

# 转换为Tensor
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

# 2. 定义Dataset类
class CoreLossDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = y.unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建Dataset实例
train_dataset = CoreLossDataset(X_train_processed, y_train_tensor)
val_dataset = CoreLossDataset(X_val_processed, y_val_tensor)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. 定义Transformer模型架构

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerRegressionModel(nn.Module):
    def __init__(self, num_numerical, num_categorical, d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=256, embedding_dim=8):
        super(TransformerRegressionModel, self).__init__()
        # 输入特征尺寸
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        
        # 数值型特征的线性层
        self.numerical_embedding = nn.Linear(num_numerical, embedding_dim)
        
        # 类别型特征的嵌入层
        self.categorical_embedding = nn.Embedding(num_categorical, embedding_dim)
        
        # Transformer编码器
        self.sequence_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 总特征尺寸 = 数值型特征 + 类别型特征 + Transformer输出
        self.fc_input_size = embedding_dim + (embedding_dim * num_categorical) + d_model
        # 回归输出层
        self.regressor = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x的尺寸：batch_size x total_features
        batch_size = x.size(0)
        # 分割输入
        numerical = x[:, :self.num_numerical]
        categorical = x[:, self.num_numerical:self.num_numerical + self.num_categorical].long()
        sequence = x[:, self.num_numerical + self.num_categorical:].unsqueeze(-1)
        
        # 数值型特征增加维度
        numerical_embedded = self.numerical_embedding(numerical)  # [batch_size, embedding_dim]
        
        # 类别型特征增加维度
        categorical_embedded = self.categorical_embedding(categorical).view(batch_size, -1)  # [batch_size, embedding_dim * num_categorical]
        
        # Transformer编码器部分
        sequence = self.sequence_embedding(sequence)  # [batch_size, seq_len, d_model]
        sequence = self.pos_encoder(sequence)

        transformer_output = self.transformer_encoder(sequence)
        transformer_output = transformer_output.mean(dim=1)  # 平均池化 [batch_size, d_model]
        
        # 合并所有特征
        combined = torch.cat([numerical_embedded, categorical_embedded, transformer_output], dim=1)

        # 回归预测
        output = self.regressor(combined)
        return output

# 4. 加载预训练模型

# 获取类别特征的数量（One-Hot编码后的维度）
num_categorical = categorical_encoded.shape[1]
# 获取数值型特征的数量
num_numerical = numerical_scaled.shape[1]

# 定义模型实例
model = TransformerRegressionModel(num_numerical=num_numerical, num_categorical=num_categorical)
model.to(device)

# 加载预训练模型参数
model.load_state_dict(torch.load('best_transformer_model.pth', weights_only=True, map_location=device))
model.eval()
print("预训练模型已加载。")

# 5. 定义传输磁能指标

def compute_transmission_energy(frequency, peak_flux_density):
    """
    计算传输磁能
    :param frequency: 赫兹（Hz）
    :param peak_flux_density: 特斯拉（T）
    :return: 传输磁能
    """
    return frequency * peak_flux_density

# 6. 定义优化问题

class MagneticCoreProblem(Problem):
    def __init__(self, model, scaler_y, onehot_encoder, scaler_num, scaler_flux, sequence_length, device):
        super().__init__(n_var=5,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([0, bounds['频率'][0], 0, bounds['磁通密度'][0], 0]),
                         xu=np.array([len(bounds['温度']) - 1, bounds['频率'][1], len(bounds['波形']) - 1, bounds['磁通密度'][1], len(bounds['磁芯材料']) - 1]),
                         type_var=np.array(["int", "real", "int", "real", "int"]))
        self.model = model
        self.scaler_y = scaler_y
        self.onehot_encoder = onehot_encoder
        self.scaler_num = scaler_num
        self.scaler_flux = scaler_flux
        self.sequence_length = sequence_length
        self.device = device

        # 定义可选的离散值列表
        self.temperature_values = bounds['温度']  # [25, 50, 70, 90]
        self.waveform_values = bounds['波形']    # [0, 1, 2]
        self.core_material_values = bounds['磁芯材料']  # [0, 1, 2, 3]

        # 获取类别特征的类别数
        self.num_core_materials = len(onehot_encoder.categories_[0])
        self.num_waveforms = len(onehot_encoder.categories_[1])

    def _evaluate(self, X, out, *args, **kwargs):
        losses = []
        energies = []
        for vars in X:
            temp_idx, frequency, waveform_idx, peak_flux_density, core_material_idx = vars

            # 将温度索引转换为实际温度值
            temp_idx = int(round(temp_idx))
            temperature = self.temperature_values[temp_idx]

            # 将波形索引和磁芯材料索引转换为整数并获取实际值
            waveform_idx = int(round(waveform_idx))
            waveform = self.waveform_values[waveform_idx]

            core_material_idx = int(round(core_material_idx))
            core_material = self.core_material_values[core_material_idx]

            # 类别特征One-Hot编码
            onehot_core = np.zeros(self.num_core_materials)
            onehot_core[core_material_idx] = 1

            onehot_waveform = np.zeros(self.num_waveforms)
            onehot_waveform[waveform_idx] = 1

            categorical_encoded = np.hstack([onehot_core, onehot_waveform])

            # 数值型特征标准化
            numerical = np.array([[temperature, frequency]])
            numerical_scaled = self.scaler_num.transform(numerical).flatten()

            # 生成磁通密度序列（假设为正弦波形）
            sequence = peak_flux_density * np.sin(np.linspace(0, 2 * np.pi, self.sequence_length))
            sequence = sequence.reshape(1, -1)
            flux_scaled = self.scaler_flux.transform(sequence).flatten()

            # 合并所有特征
            features = np.hstack([numerical_scaled, categorical_encoded, flux_scaled])

            # 转换为Tensor并移动到设备
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 预测损耗（反标准化）
            with torch.no_grad():
                loss_scaled = self.model(features_tensor).cpu().item()
            loss = self.scaler_y.inverse_transform([[loss_scaled]])[0][0]

            # 计算传输磁能
            energy = compute_transmission_energy(frequency, peak_flux_density)

            losses.append(loss)
            energies.append(energy)

        # pymoo默认是最小化，因此传输磁能需要取负数
        out["F"] = np.column_stack([losses, -np.array(energies)])

# 2. 修改算法的设置以支持整数变量

from pymoo.core.variable import Real, Integer
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

bounds = {
    '温度': [25, 50, 70, 90],          # 温度，4个取值：25、50、70、90
    '频率': [50000, 500000],   # 频率范围
    '波形': [0, 1, 2],            # 波形索引，0:正弦波, 1:三角波, 2:梯形波
    '磁通密度': [-0.305786, 0.238834],    # 磁通密度峰值范围
    '磁芯材料': [0, 1, 2, 3],        # 磁芯材料索引，0:材料1, ..., 3:材料4
}
sequence_length = 1024

# 定义变量类型
var_types = [
    Integer(bounds=(0, len(bounds['温度']) - 1)),             # 温度索引
    Real(bounds=(bounds['频率'][0], bounds['频率'][1])),     # 频率
    Integer(bounds=(0, len(bounds['波形']) - 1)),             # 波形索引
    Real(bounds=(bounds['磁通密度'][0], bounds['磁通密度'][1])), # 磁通密度
    Integer(bounds=(0, len(bounds['磁芯材料']) - 1)),           # 磁芯材料索引
]

# 创建问题实例
problem = MagneticCoreProblem(
    model=model,
    scaler_y=scaler_y,
    onehot_encoder=onehot_encoder,
    scaler_num=scaler_num,
    scaler_flux=scaler_flux,
    sequence_length=sequence_length,
    device=device
)

# 自定义采样操作，保持整数变量为整数
class MixedVariableSampling(FloatRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        # 使用随机采样生成初始种群
        X = np.random.uniform(low=problem.xl, high=problem.xu, size=(n_samples, problem.n_var))
        
        # 四舍五入处理整数变量
        X[:, 0] = np.round(X[:, 0]).astype(int)  # 温度索引
        X[:, 2] = np.round(X[:, 2]).astype(int)  # 波形索引
        X[:, 4] = np.round(X[:, 4]).astype(int)  # 磁芯材料索引
        return X

# 自定义交叉操作，确保整数变量在交叉后被四舍五入
class MixedVariableCrossover(SBX):
    def __init__(self, prob=0.7, eta=15):
        super().__init__(prob=prob, eta=eta)

    def _do(self, problem, X, **kwargs):
        offspring = super()._do(problem, X, **kwargs)
        if offspring.shape[2] != problem.n_var:
            raise ValueError(f"Unexpected number of variables in offspring: {offspring.shape[2]}, expected {problem.n_var}.")
        offspring[:, :, 0] = np.round(offspring[:, :, 0]).astype(int)  # 温度索引
        offspring[:, :, 2] = np.round(offspring[:, :, 2]).astype(int)  # 波形索引
        offspring[:, :, 4] = np.round(offspring[:, :, 4]).astype(int)  # 磁芯材料索引
        return offspring
        return offspring

# 自定义变异操作，确保整数变量在变异后被四舍五入
class MixedVariableMutation(PM):
    def __init__(self, eta=20, prob=1.0):
        super().__init__(eta=eta, prob=prob)

    def _do(self, problem, X, **kwargs):
        offspring = super()._do(problem, X, **kwargs)
        if offspring.shape[1] != problem.n_var:
            raise ValueError(f"Unexpected number of variables in mutation offspring: {offspring.shape[1]}, expected {problem.n_var}.")
        offspring[:, 0] = np.round(offspring[:, 0]).astype(int)  # 温度索引
        offspring[:, 2] = np.round(offspring[:, 2]).astype(int)  # 波形索引
        offspring[:, 4] = np.round(offspring[:, 4]).astype(int)  # 磁芯材料索引
        return offspring

# 3. 设置优化算法，使用自定义的操作以支持整数变量

algorithm = NSGA2(
    pop_size=300,
    n_offsprings=50,
    sampling=MixedVariableSampling(),
    crossover=MixedVariableCrossover(),
    mutation=MixedVariableMutation(),
    eliminate_duplicates=True
)

# 4. 执行Pareto优化

print("开始执行Pareto优化...")

res = minimize(
    problem,
    algorithm,
    ('n_gen', 50),
    verbose=True,
    seed=42
)

print("Pareto优化完成。")

# 10. 结果可视化与分析

# 提取目标函数值
losses = res.F[:, 0]
transmission_energies = -res.F[:, 1]  # 取负还原

# 提取决策变量
variables = res.X

# 创建结果DataFrame
results_df = pd.DataFrame(variables, columns=['温度（°C）', '频率（Hz）', '波形索引', '磁通密度峰值（T）', '磁芯材料索引'])
results_df['磁芯损耗（w/m3）'] = losses
results_df['传输磁能（Hz·T）'] = transmission_energies

# 将索引映射回类别名称
waveform_categories = onehot_encoder.categories_[1]
core_material_categories = onehot_encoder.categories_[0]

# 设置中文字体
font_path = '/home/zhouyh/.fonts/SimHei.ttf'
fm.fontManager.addfont(font_path)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为 SimHei
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# print("核心材料类别:", core_material_categories)
# valid_indices = results_df['磁芯材料索引'].round().astype(int).isin(range(len(core_material_categories)))
# if not valid_indices.all():
#     print("存在无效的磁芯材料索引")
# plt.figure(figsize=(8, 4))
# sns.countplot(x=results_df['磁芯材料索引'].round().astype(int))
# plt.title('磁芯材料索引分布')
# plt.xlabel('磁芯材料索引')
# plt.ylabel('数量')
# plt.show()

results_df['温度（°C）'] = results_df['温度（°C）'].round().astype(int).map(lambda x: bounds['温度'][x])
results_df['波形'] = results_df['波形索引'].round().astype(int).map(lambda x: waveform_categories[x])
results_df['磁芯材料'] = results_df['磁芯材料索引'].round().astype(int).map(lambda x: core_material_categories[x])

# 选择关键列
results_df = results_df[['温度（°C）', '频率（Hz）', '波形', '磁通密度峰值（T）', '磁芯材料', '磁芯损耗（w/m3）', '传输磁能（Hz·T）']]
# print("频率范围：", results_df['频率（Hz）'].min(), "-", results_df['频率（Hz）'].max())



# 可视化Pareto前沿
plt.figure(figsize=(10, 6))
plt.scatter(results_df['磁芯损耗（w/m3）'], results_df['传输磁能（Hz·T）'], c='red')
plt.xlabel('磁芯损耗 (w/m3)')
plt.ylabel('传输磁能 (Hz·T)')
plt.title('Pareto 前沿：最小化磁芯损耗与最大化传输磁能')
plt.grid(True)
plt.show(block=False)

# 按磁芯损耗排序，显示前10个最优解
top_solutions = results_df.sort_values('磁芯损耗（w/m3）').head(10)
print("磁芯损耗最小的10个解：")
print(top_solutions)

# 按传输磁能排序，显示前10个最优解
top_solutions_energy = results_df.sort_values('传输磁能（Hz·T）', ascending=False).head(10)
print("\n传输磁能最大的10个解：")
print(top_solutions_energy)

# 获取前160个解的索引
# 1. 对“磁芯损耗”进行排序并提取前160个解
top160_loss = results_df.sort_values('磁芯损耗（w/m3）').head(160)

# 2. 对“传输磁能”进行排序并提取前160个解
top160_energy = results_df.sort_values('传输磁能（Hz·T）', ascending=False).head(160)
top160_loss_indices = set(top160_loss.index)
top160_energy_indices = set(top160_energy.index)

# 取两个集合的交集并转换为列表
common_indices = list(top150_loss_indices.intersection(top150_energy_indices))

# 提取交集对应的解
common_solutions = results_df.loc[common_indices]
pd.set_option('display.max_rows', None)
print("同时在磁芯损耗前150个解和传输磁能前150个解中的解：")
print(common_solutions)

# 可视化各决策变量与目标的关系
df_pareto = results_df



# 散点图显示温度与磁芯损耗
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pareto, x='温度（°C）', y='磁芯损耗（w/m3）', hue='传输磁能（Hz·T）', palette='viridis')
plt.title('温度与磁芯损耗的关系')
plt.legend(title='传输磁能 (Hz·T)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# 散点图显示频率与传输磁能
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pareto, x='频率（Hz）', y='传输磁能（Hz·T）', hue='磁芯损耗（w/m3）', palette='plasma')
plt.title('频率与传输磁能的关系')
plt.legend(title='磁芯损耗 (w/m3)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# 类别型特征分析：波形
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_pareto, x='波形', y='磁芯损耗（w/m3）')
plt.title('不同波形下磁芯损耗分布')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_pareto, x='波形', y='传输磁能（Hz·T）')
plt.title('不同波形下传输磁能分布')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# 类别型特征分析：磁芯材料
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_pareto, x='磁芯材料', y='磁芯损耗（w/m3）')
plt.title('不同磁芯材料下磁芯损耗分布')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_pareto, x='磁芯材料', y='传输磁能（Hz·T）')
plt.title('不同磁芯材料下传输磁能分布')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# 聚类分析（可选）
from sklearn.cluster import KMeans

# 假设将解聚类为3类
pareto_front = res.F
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pareto_front)

# 将聚类结果加入DataFrame
df_pareto['Cluster'] = clusters

# 可视化聚类结果
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pareto, x='磁芯损耗（w/m3）', y='传输磁能（Hz·T）', hue='Cluster', palette='Set1')
plt.title('Pareto前沿聚类分析')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. 综合分析与总结

# 例如，展示哪些温度条件下损耗最小且传输能量最大
optimal_conditions = df_pareto.loc[df_pareto['传输磁能（Hz·T）'].idxmax()]
print("具有最大传输磁能的最优条件：")
print(optimal_conditions)

# 保存Pareto前沿解到Excel
df_pareto.to_excel('pareto_front_solutions.xlsx', index=False)
print("Pareto前沿的解已保存到 'pareto_front_solutions.xlsx'。")
