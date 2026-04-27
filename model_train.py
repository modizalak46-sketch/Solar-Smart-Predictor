import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Dataset load karo
try:
    df = pd.read_csv('data/solar_data.csv') 
    print("✅ Dataset loaded successfully!")
except:
    print("❌ Error: 'data/solar_data.csv' file nathi mali!")

# --- 🛠️ DATA CLEANING (Aa nava steps che error fix karva mate) ---

# A. Khali (NaN) data vala rows kadhi nakho
df = df.dropna(subset=['generation_mw', 'installedcapacity_mwp', 'datetime_gmt'])

# B. Jo koi value 'infinity' hoy to tene pan kadhi nakho
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['generation_mw'])

# --- 🛠️ DATA CLEANING END ---

# 2. Date mathi features banavo
df['datetime_gmt'] = pd.to_datetime(df['datetime_gmt'])
df['hour'] = df['datetime_gmt'].dt.hour
df['month'] = df['datetime_gmt'].dt.month

# 3. Features ane Target select karo
features = ['hour', 'month', 'installedcapacity_mwp']
target = 'generation_mw'

X = df[features]
y = df[target]

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
print("⏳ Training model... please wait.")
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 6. Save Model
with open('solar_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("🚀 Success! Model trained and 'solar_model.pkl' created.")