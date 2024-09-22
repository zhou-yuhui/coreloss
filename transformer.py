import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dataload import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

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
        position = torch.arange(
            0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerRegressionModel(nn.Module):
    def __init__(self, num_numerical, num_categorical, d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=256):
        super(TransformerRegressionModel, self).__init__()
        # 输入特征尺寸
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.d_model = d_model
        # Transformer编码器
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)
        # 总特征尺寸 = 数值型特征 + 类别型特征 + Transformer输出
        self.fc_input_size = num_numerical + num_categorical + d_model
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
        categorical = x[:, self.num_numerical:self.num_numerical + self.num_categorical]
        sequence = x[:, self.num_numerical + self.num_categorical:].unsqueeze(-1)
        # Transformer编码器部分
        sequence = self.embedding(sequence)  # [batch_size, seq_len, d_model]
        sequence = self.pos_encoder(sequence)
        sequence = sequence.permute(1, 0, 2)  # 调整维度以适应Transformer输入 [seq_len, batch_size, d_model]
        transformer_output = self.transformer_encoder(sequence)
        transformer_output = transformer_output.mean(dim=0)  # 平均池化 [batch_size, d_model]
        # 合并所有特征
        combined = torch.cat([numerical, categorical, transformer_output], dim=1)
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
model.load_state_dict(torch.load('best_transformer_model.pth'))
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

# 绘制loss曲线图
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()