import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, modcloth_path, renttherunway_path):
        self.modcloth_path = modcloth_path
        self.renttherunway_path = renttherunway_path
        
    def load_data(self):
        # 加载Modcloth数据
        modcloth_data = []
        with open(self.modcloth_path, 'r') as f:
            for line in f:
                modcloth_data.append(json.loads(line))
        modcloth_df = pd.DataFrame(modcloth_data)
        
        # 加载Rent the Runway数据
        rtr_data = []
        with open(self.renttherunway_path, 'r') as f:
            for line in f:
                rtr_data.append(json.loads(line))
        rtr_df = pd.DataFrame(rtr_data)
        
        return modcloth_df, rtr_df
    
    def preprocess_data(self, df, rating_col):
        # 获取用户和物品的唯一ID映射
        user_ids = df['user_id'].unique()
        item_ids = df['item_id'].unique()
        
        user_id_map = {id: idx for idx, id in enumerate(user_ids)}
        item_id_map = {id: idx for idx, id in enumerate(item_ids)}
        
        # 转换ID为索引
        df['user_idx'] = df['user_id'].map(user_id_map)
        df['item_idx'] = df['item_id'].map(item_id_map)
        
        # 标准化评分
        df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
        df[rating_col] = (df[rating_col] - df[rating_col].min()) / (df[rating_col].max() - df[rating_col].min())
        
        return df, len(user_ids), len(item_ids)
    
    def split_data(self, df, rating_col):
        # 准备特征和标签
        X = df[['user_idx', 'item_idx']].values
        y = df[rating_col].values
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_item_history(self, df):
        # 为每个用户获取历史交互的物品列表
        user_history = df.groupby('user_idx')['item_idx'].agg(list).to_dict()
        return user_history 