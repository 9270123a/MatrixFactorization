import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from copy import deepcopy
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, hidden_layers):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))
    
    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        
        # Concatenate user and item embeddings to form input
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        
        # Pass through MLP layers
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        
        return x

# Define a simple GMF model
class GMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
    
    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        
        # Element-wise product of user and item embeddings
        x = user_embedding * item_embedding
        
        return x

# Define NeuCF model that uses both MLP and GMF
class NeuCF(nn.Module):
    def __init__(self, num_users, num_items, mlp_embedding_size, gmf_embedding_size, hidden_layers):
        super(NeuCF, self).__init__()
        self.mlp = MLP(num_users, num_items, mlp_embedding_size, hidden_layers)
        self.gmf = GMF(num_users, num_items, gmf_embedding_size)
        self.output_layer = nn.Linear(hidden_layers[-1] + gmf_embedding_size, 1)
    
    def forward(self, user_indices, item_indices):
        mlp_output = self.mlp(user_indices, item_indices)
        gmf_output = self.gmf(user_indices, item_indices)
        
        # Concatenate outputs from MLP and GMF
        combined_output = torch.cat([mlp_output, gmf_output], dim=-1)
        
        # Final prediction
        prediction = torch.sigmoid(self.output_layer(combined_output))
        
        return prediction

# Example configuration
num_users = 50  # 根据您的实际用户数进行更新
num_items = 2090  # 根据您的实际电影数进行更新
mlp_embedding_size = 8  # Embedding size for MLP
gmf_embedding_size = 8  # Embedding size for GMF
hidden_layers = [16, 32, 16, 8]  # Example hidden layers configuration for MLP

# Initialize the NeuCF model with the example configuration
neucf_model = NeuCF(num_users, num_items, mlp_embedding_size, gmf_embedding_size, hidden_layers)

# 注意：请根据实际情况修改路径
interaction_matrix_path = r"C:\AI專案\MatrixFactorization\MovieLean\interaction_matrix_full.csv"
interaction_matrix = pd.read_csv(interaction_matrix_path, index_col='userId')


print(interaction_matrix.shape)

class RatingDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]




# Prepare the data for training
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    best_val_rmse = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            user_indices, item_indices, labels = batch
            optimizer.zero_grad()
            outputs = model(user_indices, item_indices)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                user_indices, item_indices, labels = batch
                preds = model(user_indices, item_indices)
                all_preds.extend(preds.squeeze().tolist())
                all_labels.extend(labels.tolist())

        val_rmse = sqrt(mean_squared_error(all_labels, all_preds))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = deepcopy(model.state_dict())

    return best_val_rmse, best_model_state
def load_and_prepare_dataset(csv_path, test_size=0.2):
    # 加载CSV文件
    df = pd.read_csv(csv_path, index_col='userId')
    
    # 初始化列表来存储用户ID、项目ID和评分
    user_ids = []
    item_ids = []
    ratings = []
    
    # 遍历DataFrame以构建用户ID、项目ID和评分的列表
    for user_id, row in df.iterrows():
        for item_id, rating in enumerate(row):
            if rating > 0:  # 如果评分大于0，则用户对该项目有评分
                user_ids.append(user_id)
                item_ids.append(item_id)
                ratings.append(rating)  # 这里假设评分是二进制的（例如1表示喜欢）

    # 将列表转换为Tensor
    user_ids_tensor = torch.tensor(user_ids, dtype=torch.long)
    item_ids_tensor = torch.tensor(item_ids, dtype=torch.long)
    ratings_tensor = torch.tensor(ratings, dtype=torch.float32)
    
    # 分割数据集为训练集和验证集
    dataset = TensorDataset(user_ids_tensor, item_ids_tensor, ratings_tensor)
    num_samples = len(dataset)
    train_size = int((1 - test_size) * num_samples)
    val_size = num_samples - train_size
    
    # 使用random_split来分割TensorDataset
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader



# 参数网格



param_grid = {
    'batch_size': [128, 256, 512],
    'lr': [0.0001, 0.001, 0.01],
    'embedding_size': [8, 16, 32]
}

best_loss = float('inf')
best_settings = None

# 遍历参数网格
for batch_size in param_grid['batch_size']:
    for lr in param_grid['lr']:
        for embedding_size in param_grid['embedding_size']:
            print(f"Training with batch_size={batch_size}, lr={lr}, embedding_size={embedding_size}")
            
            # 加载和准备数据集，确保传入正确的CSV文件路径
            train_loader, val_loader = load_and_prepare_dataset(interaction_matrix_path, test_size=0.2)  # 注意这里传入的是文件路径和可选的test_size参数
            
            # 初始化模型
            model = NeuCF(num_users, num_items, embedding_size, embedding_size, [embedding_size*2, embedding_size, embedding_size//2])
            
            # 设置优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # 训练和评估模型
            val_loss, model_state = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer)

            # 更新最佳模型（如果适用）
            if val_loss < best_loss:
                best_loss = val_loss
                best_settings = {'batch_size': batch_size, 'lr': lr, 'embedding_size': embedding_size}

best_settings_dict = {
    'best_loss': best_loss,
    'best_batch_size': best_settings['batch_size'],
    'best_lr': best_settings['lr'],
    'best_embedding_size': best_settings['embedding_size']
}

# 使用torch.save来保存这个字典
torch.save(best_settings_dict, 'best_settings.pth')
print(best_settings_dict)


