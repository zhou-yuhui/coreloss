import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pickle

# 设置设备为 GPU（如果可用）否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 定义与训练时相同的 TransformerRegressionModel 类
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

# 定义自定义 Dataset
class CoreLossTestDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

def main():
    # 1. 加载预处理转换器
    with open('onehot_encoder.pkl', 'rb') as f:
        onehot_encoder = pickle.load(f)
    with open('scaler_num.pkl', 'rb') as f:
        scaler_num = pickle.load(f)
    with open('scaler_flux.pkl', 'rb') as f:
        scaler_flux = pickle.load(f)
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    # 2. 读取测试集数据
    test_df = pd.read_excel('testset.xlsx')
    
    # 3. 提取特征
    numerical_features = ['温度，oC', '频率，Hz']
    categorical_features = ['磁芯材料', '励磁波形']
    sequence_features = test_df.columns[5:]  # 假设前5列为：序号、温度，oC、频率，Hz、磁芯材料、励磁波形
    
    X_test = test_df[numerical_features + categorical_features + list(sequence_features)]
    
    # 4. 预处理数值型特征
    numerical_scaled = scaler_num.transform(X_test[numerical_features])
    
    # 5. 预处理类别型特征
    categorical_encoded = onehot_encoder.transform(X_test[categorical_features])
    
    # 6. 预处理序列特征（磁通密度）
    flux_density = X_test[sequence_features].values
    flux_density_scaled = scaler_flux.transform(flux_density)
    
    # 7. 合并所有处理后的特征
    X_test_processed = np.hstack([numerical_scaled, categorical_encoded, flux_density_scaled])
    
    # 8. 创建测试集的 DataLoader
    test_dataset = CoreLossTestDataset(X_test_processed)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 9. 初始化模型
    num_categorical = categorical_encoded.shape[1] # 计算每个类别特征的编码维度
    num_numerical = numerical_scaled.shape[1]

    # num_categorical = 7
    # num_numerical = 2
    model = TransformerRegressionModel(num_numerical=num_numerical, num_categorical=num_categorical)
    
    # 10. 加载模型参数
    model.load_state_dict(torch.load('best_transformer_model.pth', weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    
    # 11. 进行预测
    predictions = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().squeeze().numpy())
    
    # 12. 逆转换预测结果
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_inverse = scaler_y.inverse_transform(predictions).flatten()
    
    # 13. 保留小数点后1位
    predictions_rounded = np.round(predictions_inverse, 1)
    
    # 14. 将预测结果插入 result.xlsx
    result_df = pd.read_excel('result.xlsx')
    
    # 确保 '序号' 列存在并匹配
    if '序号' not in result_df.columns:
        raise ValueError("result.xlsx 中缺少 '序号' 列。")
    
    # 创建一个映射从序号到索引
    seq_to_index = {seq: idx for idx, seq in enumerate(result_df['序号'])}
    
    # 遍历测试集中的每一行，将预测结果填入相应的序号
    for i, row in test_df.iterrows():
        seq = row['序号']
        if seq in seq_to_index:
            result_df.at[seq_to_index[seq], result_df.columns[2]] = predictions_rounded[i]
        else:
            print(f"序号 {seq} 在 result.xlsx 中未找到。")
    
    # 15. 保存 result.xlsx，保留原文件名
    result_df.to_excel('result.xlsx', index=False)
    print("预测结果已保存到 result.xlsx 中。")
    
    # 16. 打印特定序号的预测结果
    target_seqs = [16, 76, 98, 126, 168, 230, 271, 338, 348, 379]
    print("特定样本的磁芯损耗预测结果：")
    for seq in target_seqs:
        if seq in seq_to_index:
            predicted_loss = result_df.at[seq_to_index[seq], result_df.columns[2]]
            print(f"序号 {seq}: {predicted_loss:.1f} w/m³")
        else:
            print(f"序号 {seq} 在 result.xlsx 中未找到。")

if __name__ == "__main__":
    main()
