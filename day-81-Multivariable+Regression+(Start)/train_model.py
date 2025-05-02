# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# 1. Load data
df = pd.read_csv('boston.csv', index_col=0)

# 2. Pisahkan fitur dan target
X = df.drop(columns='PRICE')  #target (harga rumah)
y = np.log(df['PRICE'])

# 3. Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Buat dan latih model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Simpan model ke file .pkl
joblib.dump(model, 'model.pkl')

print("Model berhasil dilatih dan disimpan sebagai model.pkl")
