import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 全局配置
DATASET_CONFIG = {
    'modcloth': {
        'input_path': '../Data_full/modcloth_final_data.json',
        'output_path': '../Data_full/modcloth_final_data',
        'categorical_cols': ['category', 'fit', 'length', 'cup size', 'bra size'],
        'numeric_cols': ['size', 'quality', 'waist', 'hips', 'bust']
    },
    'renttherunway': {
        'input_path': '../Data_full/renttherunway_final_data.json',
        'output_path': '../Data_full/renttherunway_final_data',
        'categorical_cols': ['category', 'fit', 'body type', 'rented for'],
        'numeric_cols': ['size', 'rating', 'age', 'weight', 'bust size', 'waist size', 'hips size']
    }
}

# 当前使用的数据集名称
CURRENT_DATASET = 'modcloth'  # 可以切换为 'renttherunway'

def load_data(file_path):
    """加载JSON数据文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

def convert_weight_to_lbs(weight_str):
    """将体重字符串转换为磅值"""
    if pd.isna(weight_str) or weight_str == 'nan':
        return None
    try:
        # 移除所有空格并转换为小写
        weight_str = str(weight_str).lower().strip()
        
        # 处理不同的格式
        if 'lbs' in weight_str:
            return float(weight_str.replace('lbs', '').strip())
        elif 'kg' in weight_str:
            kg = float(weight_str.replace('kg', '').strip())
            return round(kg * 2.20462)  # 将千克转换为磅
        elif weight_str.replace('.', '').isdigit():
            weight = float(weight_str)
            if weight > 500:  # 假设大于500的是克
                return round(weight / 453.592)  # 将克转换为磅
            elif weight > 200:  # 假设大于200的是斤
                return round(weight * 1.10231)  # 将斤转换为磅
            else:
                return weight  # 假设已经是磅
        else:
            return None
    except (ValueError, AttributeError, TypeError):
        return None

def preprocess_data(df, dataset_type):
    """数据预处理主函数"""
    # 1. 删除重复行
    df = df.drop_duplicates()
    
    config = DATASET_CONFIG[dataset_type]
    categorical_cols = config['categorical_cols']
    numeric_cols = config['numeric_cols']
    
    # 2. 数据类型转换和清理
    # 确保所有文本列都是字符串类型，同时正确处理None值
    text_cols = ['review_text', 'review_summary', 'height', 'category', 'fit', 'body type', 'rented for', 'weight']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) and x is not None else np.nan)
            df[col] = df[col].replace('nan', np.nan).replace('None', np.nan)
    
    # 3. 处理特殊字段
    # 处理体重数据
    if 'weight' in df.columns:
        df['weight'] = df['weight'].replace('nan', np.nan)
        df['weight_lbs'] = df['weight'].apply(convert_weight_to_lbs)
        # 使用体重的中位数填充缺失值
        median_weight = df['weight_lbs'].median()
        df['weight_lbs'] = df['weight_lbs'].fillna(median_weight)
        # 更新weight列
        df['weight'] = df['weight_lbs']
    
    # 处理身高数据
    if 'height' in df.columns:
        df['height'] = df['height'].replace('nan', np.nan)
        df['height_inches'] = df['height'].apply(convert_height_to_inches)
        # 使用身高的中位数填充缺失值
        median_height = df['height_inches'].median()
        df['height_inches'] = df['height_inches'].fillna(median_height)
    
    # 4. 处理缺失值
    # 对于分类特征使用众数填充
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 对于数值特征使用中位数填充
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # 5. 标准化分类变量
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
    
    # 6. 处理异常值
    for col in numeric_cols:
        if col in df.columns:
            # 使用IQR方法处理异常值
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
    
    # 7. 创建新特征
    if dataset_type == 'modcloth':
        df['has_review'] = df['review_text'].notna().astype(int)
        if 'review_text' in df.columns:
            df['review_length'] = df['review_text'].fillna('').str.len()
    elif dataset_type == 'renttherunway':
        # renttherunway特有的特征处理
        if 'review_text' in df.columns:
            df['review_sentiment'] = df['review_text'].apply(analyze_sentiment)
        if 'rented for' in df.columns:
            df = pd.get_dummies(df, columns=['rented for'], prefix='rented_for')
        
        # 计算BMI
        if 'weight' in df.columns and 'height_inches' in df.columns:
            df['bmi'] = calculate_bmi(df['weight'], df['height_inches'])
    
    return df

def analyze_sentiment(text):
    """简单的情感分析"""
    if pd.isna(text) or text == 'nan' or text is None or not isinstance(text, str):
        return 0
    
    text = str(text).lower()
    positive_words = ['love', 'great', 'perfect', 'amazing', 'excellent', 'beautiful', 
                     'comfortable', 'fantastic', 'wonderful', 'happy', 'best']
    negative_words = ['bad', 'poor', 'terrible', 'horrible', 'disappointed', 'worst',
                     'uncomfortable', 'hate', 'awful', 'wrong']
    
    score = 0
    for word in positive_words:
        if word in text:
            score += 1
    for word in negative_words:
        if word in text:
            score -= 1
    return score

def calculate_bmi(weight, height_inches):
    """计算BMI"""
    # BMI = (weight in pounds * 703) / (height in inches)²
    return (weight * 703) / (height_inches ** 2)

def convert_height_to_inches(height_str):
    """将身高字符串转换为英寸数值"""
    if pd.isna(height_str):
        return None
        
    try:
        # 移除所有空格并转换为小写
        height_str = height_str.lower().strip()
        
        # 处理不同的格式
        if "'" in height_str:  # 处理 5'7" 格式
            feet, inches = height_str.replace('"', '').split("'")
            return int(feet) * 12 + int(inches)
        elif 'ft' in height_str:  # 处理 5ft 7in 格式
            parts = height_str.replace('ft', '').replace('in', '').split()
            if len(parts) == 2:
                return int(parts[0]) * 12 + int(parts[1])
            else:
                return int(parts[0]) * 12
        elif 'cm' in height_str:  # 处理厘米格式
            cm = float(height_str.replace('cm', '').strip())
            return round(cm / 2.54)  # 将厘米转换为英寸
        elif height_str.replace('.', '').isdigit():  # 处理纯数字格式
            if float(height_str) > 12:  # 假设大于12的是厘米
                return round(float(height_str) / 2.54)
            else:
                return int(float(height_str) * 12)  # 假设是英尺
        else:
            return None
    except (ValueError, AttributeError, TypeError):
        return None

def save_processed_data(df, output_path):
    """保存处理后的数据"""
    # 保存为CSV格式
    df.to_csv(output_path + '_processed.csv', index=False)
    
    # 保存为JSON格式
    df.to_json(output_path + '_processed.json', orient='records', lines=True)

def main():
    """主函数"""
    config = DATASET_CONFIG[CURRENT_DATASET]
    
    # 加载数据
    print(f"Loading {CURRENT_DATASET} dataset...")
    df = load_data(config['input_path'])
    
    # 预处理数据
    print("Preprocessing data...")
    processed_df = preprocess_data(df, CURRENT_DATASET)
    
    # 保存处理后的数据
    print("Saving processed data...")
    save_processed_data(processed_df, config['output_path'])
    
    print("Data preprocessing completed!")
    
    # 打印数据集基本信息
    print("\nDataset Info:")
    print(f"Total samples: {len(processed_df)}")
    print("\nFeature statistics:")
    print(processed_df.describe())
    
    # 打印特征列表
    print("\nFeatures in processed dataset:")
    print(processed_df.columns.tolist())

if __name__ == "__main__":
    main()
