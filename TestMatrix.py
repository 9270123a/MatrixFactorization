import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score

# Define a simple MLP model
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
        
        # Concatenate outputs from M0P and GMF
        combined_output = torch.cat([mlp_output, gmf_output], dim=-1)
        
        # Final prediction
        prediction = torch.sigmoid(self.output_layer(combined_output))
        
        return prediction


def predict_interaction(model, user_id, movie_id, interaction, threshold=0.6): # 预测

    movie_id = int(movie_id)  # 将字符串转换为整数

    # 将user_id和movie_id转换为适合模型输入的tensor格式
    user_tensor = torch.tensor([user_id], dtype=torch.long)
    movie_tensor = torch.tensor([movie_id], dtype=torch.long)

    # 不计算梯度以加速和节省内存
    with torch.no_grad():
        # 获取模型对当前用户和物品交互的预测得分
        prediction_score = model(user_tensor, movie_tensor).item()
        # 如果预测得分大于阈值，则认为是正向交互（例如，用户喜欢该物品）
        predicted_interaction = 1 if prediction_score >= threshold else 0

    # 返回预测的交互、是否正确的标志和预测得分
    return predicted_interaction, (predicted_interaction == interaction), prediction_score


# 模型配置参数
num_users = 50  # 根据你的数据集调整
num_items = 2079  # 根据你的数据集调整
mlp_embedding_size = 8
gmf_embedding_size = 8
hidden_layers = [16, 8, 8, 8]

# 初始化NeuCF模型
neucf_model = NeuCF(num_users, num_items, mlp_embedding_size, gmf_embedding_size, hidden_layers)

# 加载训练过的模型参数
model_path = r'C:\AI專案\MatrixFactorization\neucf_model.pth'
neucf_model.load_state_dict(torch.load(model_path))

# 将模型设置为评估模式
neucf_model.eval()

# 加载测试数据集
test_dataset_path = r"C:\AI專案\MatrixFactorization\MovieLean\interaction_matrix_test.csv"
test_matrix = pd.read_csv(test_dataset_path, index_col=0)

# 初始化列表用于存储预测的交互和实际交互
predicted_interactions = []
actual_interactions = []

# 遍历测试矩阵并进行预测
for user_id, row in enumerate(test_matrix.itertuples(index=False), 0):
    for movie_id, interaction in enumerate(row, 0):
        actual_movie_id = test_matrix.columns[movie_id]
        predicted_interaction, correct_prediction, prediction_score = predict_interaction(neucf_model, user_id, actual_movie_id, interaction, 0.9)
        predicted_interactions.append(predicted_interaction)
        actual_interactions.append(interaction)
        
        # 打印用户ID、物品ID、预测的交互、真实的交互以及预测得分
        print(f"UserID: {user_id}, ItemID: {actual_movie_id}, Predicted Interaction: {predicted_interaction}, Actual Interaction: {interaction}, Prediction Score: {prediction_score}")

        
# 计算准确率
accuracy = accuracy_score(actual_interactions, predicted_interactions)
print(f'Accuracy: {accuracy}')
