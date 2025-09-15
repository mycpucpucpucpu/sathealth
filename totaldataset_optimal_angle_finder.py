import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

# === Parameter Settings ===
data_path = './data/data_combined_new.xlsx'
output_dir = './outputs/'
output_file = os.path.join(output_dir, 'results.xlsx')
os.makedirs(output_dir, exist_ok=True)

label = 'current'

# === 1. Read dataset ===
df = pd.read_excel(data_path)

# === 2. Division ===
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# === 3. Standardize and save the mean and standard deviation ===
scaler = StandardScaler()
train_df[label] = scaler.fit_transform(train_df[[label]])
val_df[label] = scaler.transform(val_df[[label]])

mean_current, std_current = scaler.mean_[0], scaler.scale_[0]

scale_params_file = os.path.join(output_dir, 'scale_params.json')
if not os.path.exists(scale_params_file):
    with open(scale_params_file, 'w') as f:
        json.dump({'mean_current': mean_current, 'std_current': std_current}, f)

# === 4. Convert to AutoGluon data format ===
train_data = TabularDataset(train_df)
val_data = TabularDataset(val_df)

# === 5. Load and train ===
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


#=== 6. Optimal angle predict function ===
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

# === 7. Predict ===
results = []
for _, row in df.iterrows():
    best_angle, max_current = find_max_current_angle_vectorized(row.copy(), predictor, mean_current, std_current)
    results.append({
        **row.to_dict(),
        'max_current': max_current,
        'best_angle': best_angle
    })

# === 8. Save results ===
df_out = pd.DataFrame(results)
df_out.to_excel(output_file, index=False)
print(f" saved to：{output_file}")
