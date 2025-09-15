import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

# === 参数设置 ===
data_path = './data/data_combined_new.xlsx'
output_dir = './outputs/'
output_file = os.path.join(output_dir, 'results.xlsx')
os.makedirs(output_dir, exist_ok=True)

label = '电流'

# === 1. 读取完整数据集并进行处理 ===
df = pd.read_excel(data_path)

# === 2. 拆分训练集/验证集 ===
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# === 3. 标准化和保存均值、标准差用于反归一化 ===
scaler = StandardScaler()
train_df[label] = scaler.fit_transform(train_df[[label]])
val_df[label] = scaler.transform(val_df[[label]])

mean_current, std_current = scaler.mean_[0], scaler.scale_[0]

# 检查如果是第一次运行，则需要创建和保存标准化参数
scale_params_file = os.path.join(output_dir, 'scale_params.json')
if not os.path.exists(scale_params_file):
    with open(scale_params_file, 'w') as f:
        json.dump({'mean_current': mean_current, 'std_current': std_current}, f)

# === 4. 转换为 AutoGluon 数据格式 ===
train_data = TabularDataset(train_df)
val_data = TabularDataset(val_df)

# === 5. 加载或训练模型 ===
model_dir = 'models/current_prediction_scaled'
predictor_path = os.path.join(model_dir, 'predictor.pkl')

if os.path.exists(predictor_path):
    predictor = TabularPredictor.load(model_dir)
else:
    os.makedirs(model_dir, exist_ok=True)
    predictor = TabularPredictor(label=label, path=model_dir).fit(
        train_data=train_data,
        tuning_data=val_data, 
        use_bag_holdout=True, 
        time_limit=600,
        presets='high_quality',
        hyperparameters={'GBM': {}}
    )


#=== 5. 预测最大电流角度（向量化实现）===
def find_max_current_angle_vectorized(row, predictor, mean_current, std_current):
    angles = np.arange(1, 361)
    angle_rad = np.radians(angles)
    angle_sin = np.sin(angle_rad)
    angle_cos = np.cos(angle_rad)

    repeated = pd.DataFrame([row.to_dict()] * 360)
    repeated['angle_sin'] = angle_sin
    repeated['angle_cos'] = angle_cos

    pred_data = TabularDataset(repeated)
    current_scaled = predictor.predict(pred_data).values
    current = current_scaled * std_current + mean_current

    max_idx = np.argmax(current)
    return angles[max_idx], current[max_idx]

# === 6. 遍历每行数据进行预测 ===
results = []
for _, row in df.iterrows():
    best_angle, max_current = find_max_current_angle_vectorized(row.copy(), predictor, mean_current, std_current)
    results.append({
        **row.to_dict(),
        'max_current': max_current,
        'best_angle': best_angle
    })

# === 7. 保存结果 ===
df_out = pd.DataFrame(results)
df_out.to_excel(output_file, index=False)
print(f" 预测完成，已保存至：{output_file}")
