import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 读取Excel文件中的每个工作表
# sheet1 = pd.read_excel('trainingset.xlsx', sheet_name='材料1', nrows=100)
# sheet2 = pd.read_excel('trainingset.xlsx', sheet_name='材料2', nrows=100)
# sheet3 = pd.read_excel('trainingset.xlsx', sheet_name='材料3', nrows=100)
# sheet4 = pd.read_excel('trainingset.xlsx', sheet_name='材料4', nrows=100)
sheet1 = pd.read_excel('trainingset.xlsx', sheet_name='材料1')
sheet2 = pd.read_excel('trainingset.xlsx', sheet_name='材料2')
sheet3 = pd.read_excel('trainingset.xlsx', sheet_name='材料3')
sheet4 = pd.read_excel('trainingset.xlsx', sheet_name='材料4')
sheet1.insert(3, '磁芯材料', '材料1')
sheet2.insert(3, '磁芯材料', '材料2')
sheet3.insert(3, '磁芯材料', '材料3')
sheet4.insert(3, '磁芯材料', '材料4')

# 合并所有工作表
data = pd.concat([sheet1, sheet2, sheet3, sheet4], ignore_index=True)

# 提取特征和目标变量
# 数值型特征
numerical_features = ['温度，oC', '频率，Hz']
# 类别型特征
categorical_features = ['磁芯材料', '励磁波形']
# 序列特征（磁通密度）
sequence_features = data.columns[5:]

X = data[numerical_features + categorical_features + list(sequence_features)]
# 目标变量
y = data['磁芯损耗，w/m3'].values

# 拆分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 对类别特征进行One-Hot编码
onehot_encoder = OneHotEncoder(sparse_output=False)
categorical_encoded = onehot_encoder.fit_transform(X_train[categorical_features])
X_val_categorical_encoded = onehot_encoder.transform(X_val[categorical_features])

# 对数值特征进行标准化
scaler_num = MinMaxScaler()
numerical_scaled = scaler_num.fit_transform(X_train[numerical_features])
X_val_numerical_scaled = scaler_num.transform(X_val[numerical_features])

# 对磁通密度序列进行标准化
scaler_flux = MinMaxScaler()
flux_density = X_train[sequence_features].values
flux_density_scaled = scaler_flux.fit_transform(flux_density)
X_val_flux_density_scaled = scaler_flux.transform(X_val[sequence_features].values)
# # 合并所有处理后的特征
X_train_processed = np.hstack([numerical_scaled, categorical_encoded, flux_density_scaled])
X_val_processed = np.hstack([X_val_numerical_scaled, X_val_categorical_encoded, X_val_flux_density_scaled])

# 初始化 StandardScaler
scaler_y = StandardScaler()

# 拟合并转换训练集的目标变量
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# 仅转换验证集的目标变量
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

# 将标签转换为Tensor
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

np.set_printoptions(threshold=np.inf)  # 设置打印选项为无限制，防止省略
# 检查合并后特征的维度
# print(X_train_processed[0])  # 确认特征数量是否正确

# # 将励磁波形类型转换为数值
# data['励磁波形'] = data['励磁波形'].map({'正弦波': 0, '三角波': 1, '梯形波': 2})
# data['磁芯材料'] = data['磁芯材料'].map({'材料1': 0, '材料2': 1, '材料3': 2, '材料4': 3})
# import pickle

# # 在训练脚本的最后部分添加
# with open('onehot_encoder.pkl', 'wb') as f:
#     pickle.dump(onehot_encoder, f)

# with open('scaler_num.pkl', 'wb') as f:
#     pickle.dump(scaler_num, f)

# with open('scaler_flux.pkl', 'wb') as f:
#     pickle.dump(scaler_flux, f)

# with open('scaler_y.pkl', 'wb') as f:
#     pickle.dump(scaler_y, f)