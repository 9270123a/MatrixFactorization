import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

input_excel_path = r"C:\AI專案\MatrixFactorization\MovieLean\rating.xlsx"
output_train_csv_path = r"C:\AI專案\MatrixFactorization\MovieLean\interaction_matrix_train.csv"
output_test_csv_path = r"C:\AI專案\MatrixFactorization\MovieLean\test.csv"

# 加载数据
ratings = pd.read_excel(input_excel_path, engine='openpyxl')

# 随机切分原始数据为两半
ratings_half_1, ratings_half_2 = train_test_split(ratings, test_size=0.5, random_state=42)

# 规范化ID
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

ratings_half_1['userId'] = user_encoder.fit_transform(ratings_half_1['userId'])
ratings_half_1['movieId'] = item_encoder.fit_transform(ratings_half_1['movieId'])

# 二值化评分
ratings_half_1['rating'] = (ratings_half_1['rating'] > 0).astype(int)

# 留一法 - 仅应用于一半数据
def train_test_split_leave_one_out(ratings_df):
    test_records = []
    train_records = ratings_df.copy()
    for user_id in ratings_df['userId'].unique():
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        test_record = user_ratings.sample(1, random_state=42)  # 随机选择一个作为测试集
        test_records.append(test_record)
        train_records = train_records.drop(test_record.index)  # 从训练记录中移除测试记录
    
    test_df = pd.concat(test_records).reset_index(drop=True)
    return train_records, test_df

train_df, test_df = train_test_split_leave_one_out(ratings_half_1)

# 创建训练集的用户-物品交互矩阵
train_interaction_matrix = train_df.pivot(index='userId', columns='movieId', values='rating').fillna(0).astype(int)

# 保存训练集交互矩阵为CSV文件
train_interaction_matrix.to_csv(output_train_csv_path)

# 保存另一半数据作为独立的测试集CSV文件
ratings_half_2.to_csv(output_test_csv_path, index=False)

print(f"Train interaction matrix saved to {output_train_csv_path}")
print(f"Independent test dataset saved to {output_test_csv_path}")
