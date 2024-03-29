import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#指定gpu跑
print("Using device:", device)

class MLP(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, hidden_layers):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size).to(device)
        self.item_embedding = nn.Embedding(num_items, embedding_size).to(device)
        self.fc_layers = nn.ModuleList([nn.Linear(in_size, out_size).to(device) for in_size, out_size in zip(hidden_layers[:-1], hidden_layers[1:])])
    
    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        return x

class GMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size).to(device)
        self.item_embedding = nn.Embedding(num_items, embedding_size).to(device)
    
    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        x = user_embedding * item_embedding
        return x

class NeuCF(nn.Module):
    def __init__(self, num_users, num_items, mlp_embedding_size, gmf_embedding_size, hidden_layers):
        super(NeuCF, self).__init__()
        self.mlp = MLP(num_users, num_items, mlp_embedding_size, hidden_layers).to(device)
        self.gmf = GMF(num_users, num_items, gmf_embedding_size).to(device)
        self.output_layer = nn.Linear(hidden_layers[-1] + gmf_embedding_size, 1).to(device)
    
    def forward(self, user_indices, item_indices):
        mlp_output = self.mlp(user_indices, item_indices)
        gmf_output = self.gmf(user_indices, item_indices)
        combined_output = torch.cat([mlp_output, gmf_output], dim=-1)
        prediction = torch.sigmoid(self.output_layer(combined_output))
        return prediction

# 根据您的具体需求更新以下配置
num_users = 50
num_items = 2079
mlp_embedding_size = 8
gmf_embedding_size = 8
hidden_layers = [16, 8, 8, 8]

neucf_model = NeuCF(num_users, num_items, mlp_embedding_size, gmf_embedding_size, hidden_layers).to(device)

interaction_matrix_path = r"C:\AI專案\MatrixFactorization\MovieLean\interaction_matrix_train.csv"
interaction_matrix = pd.read_csv(interaction_matrix_path)

print(interaction_matrix.shape)

class RatingDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        # 如果user_ids, item_ids, 和 ratings 已经是张量了，就使用.clone().detach()来避免警告
        self.user_ids = user_ids.clone().detach().to(device)
        self.item_ids = item_ids.clone().detach().to(device)
        self.ratings = ratings.clone().detach().to(device)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


# Extracting user-item interactions from the matrix
user_ids = []
item_ids = []
ratings = []
neucf_model.to(device)
for user_id in range(num_users):
    for item_id in range(num_items):
        rating = interaction_matrix.iloc[user_id, item_id + 1]  # +1 to skip the 'userId' column
        # if rating > 0:
        user_ids.append(user_id)
        item_ids.append(item_id)
        ratings.append(rating)  # 此处rating都为1，表示已评分

# Convert lists to tensors
user_ids_tensor = torch.tensor(user_ids, dtype=torch.long)
item_ids_tensor = torch.tensor(item_ids, dtype=torch.long)
ratings_tensor = torch.tensor(ratings, dtype=torch.float32)

# Create a Dataset and DataLoader for training
dataset = RatingDataset(user_ids_tensor, item_ids_tensor, ratings_tensor)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)


# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(neucf_model.parameters(), lr=0.0001)

# Train the model
num_epochs = 5
print_every = 100 # 每100个batch打印一次
losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (user_indices, item_indices, labels) in enumerate(data_loader, 0):
        # 前向传播
        outputs = neucf_model(user_indices, item_indices).squeeze()
        loss = criterion(outputs, labels)

        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # 计算每个epoch的平均损失并收集
    epoch_loss = running_loss / len(data_loader)
    losses.append(epoch_loss)
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

# 训练结束后，绘制损失图
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
print('Training complete')
# Save the model
torch.save(neucf_model.state_dict(), 'neucf_model.pth')
# Save the model









