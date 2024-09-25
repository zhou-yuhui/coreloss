import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dataload import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib

# 设置设备为 GPU（如果可用）否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

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
        categorical_embedded = self.categorical_embedding(categorical).view(batch_size, -1)  # [batch_size, embedding_dim]
        
        # Transformer编码器部分
        sequence = self.sequence_embedding(sequence)  # [batch_size, seq_len, d_model]
        sequence = self.pos_encoder(sequence)

        transformer_output = self.transformer_encoder(sequence)
        transformer_output = transformer_output.mean(dim=1)  # 平均池化 [batch_size, d_model]
        
        # 合并所有特征
        combined = torch.cat([numerical_embedded, categorical_embedded, transformer_output], dim=1)

        # print(f"numerical_embedded: {numerical_embedded.shape}")
        # print(f"categorical_embedded: {categorical_embedded.shape}")
        # print(f"transformer_output: {transformer_output.shape}")
        # print(f"combined: {combined.shape}")

        # 回归预测
        output = self.regressor(combined)
        return output

# 获取类别型特征的数量
num_categorical = categorical_encoded.shape[1]
# 获取数值型特征的数量
num_numerical = numerical_scaled.shape[1]
# 定义模型
model = TransformerRegressionModel(num_numerical=num_numerical, num_categorical=num_categorical)
# print(model)
# exit()
model.to(device)  # 将模型移动到设备上

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)  # 将输入数据移动到设备上
        batch_y = batch_y.to(device)  # 将标签数据移动到设备上

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)  # 将输入数据移动到设备上
            batch_y = batch_y.to(device)  # 将标签数据移动到设备上

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    # 打印训练过程
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # 保存最好的模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_transformer_model.pth')

# 加载最佳模型
model.load_state_dict(torch.load('best_transformer_model.pth', weights_only=True))
model.to(device)  # 确保模型在设备上
model.eval()

# 在验证集上评估模型性能
val_predictions = []
val_targets = []
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)  # 将输入数据移动到设备上
        outputs = model(batch_x)
        val_predictions.extend(outputs.cpu().squeeze().numpy())  # 将输出移回CPU
        val_targets.extend(batch_y.squeeze().numpy())

# 计算评估指标，如均方误差（MSE）和平均绝对误差（MAE）
mse = mean_squared_error(val_targets, val_predictions)
mae = mean_absolute_error(val_targets, val_predictions)

print(f'Validation MSE: {mse:.4f}, MAE: {mae:.4f}')

# 设置中文字体
font_path = '/home/zhouyh/.fonts/SimHei.ttf'
fm.fontManager.addfont(font_path)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为 SimHei
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# 绘制loss曲线图
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.show(block=False)

# 绘制预测值与实际值的散点图
plt.figure(figsize=(8, 8))
plt.scatter(val_targets, val_predictions, alpha=0.6)
plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], 'r--')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('预测值与实际值的散点图')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# 计算残差
residuals = np.array(val_targets) - np.array(val_predictions)

# 绘制残差的直方图
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('残差值')
plt.ylabel('频数')
plt.title('残差分布直方图')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# 绘制残差与预测值的散点图
plt.figure(figsize=(8, 6))
plt.scatter(val_predictions, residuals, alpha=0.6)
plt.hlines(y=0, xmin=min(val_predictions), xmax=max(val_predictions), colors='r', linestyles='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差与预测值的关系图')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

numerical_features = ['温度，oC', '频率，Hz']  # 替换为您的数值型特征名称列表

for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature], df['目标变量'], alpha=0.6)
    plt.xlabel(feature)
    plt.ylabel('目标变量')
    plt.title(f'{feature} 与目标变量的关系图')
    plt.grid(True)
    plt.tight_layout()
    plt.show()