import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import numpy as np
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt 

ticker = 'AAPL' 
date = '2010-01-01'
length2 = [5,10,15,20,25,50]

# 1. Download Data (Using the fix to avoid the MultiIndex Error)
df = yf.download(tickers=ticker, start=date)

df.columns = df.columns.get_level_values(0)

# 2. Indicators - These will now work!
df.ta.macd(append=True) 
df.ta.bbands(append=True)

for i in length2:
    df[f'SMA_{i}'] = ta.sma(df['Close'], length=i)
    df[f'EMA_{i}'] = ta.ema(df['Close'], length=i)
    df[f'RSI_{i}'] = ta.rsi(df['Close'], length=i)
    df[f'ATR_{i}'] = ta.atr(df['High'], df['Low'], df['Close'], length=i)

# 3. Create Target (Predicting the next day's PERCENTAGE CHANGE) & Clean
df['Target'] = df['Close'].pct_change().shift(-1) # <-- Predicting Returns
df.dropna(inplace=True)

# 4. Feature Selection & Train/Test Split
features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target']]
X = df[features]
y = df['Target']

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Scaling (Required for NN, SVR, Linear)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train All Models (Using your optimized RF params)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05)
lr_model = LinearRegression()
# OPTIMIZED RANDOM FOREST
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42)
svr_model = SVR(kernel='rbf') # SVR needs tuning if performance is poor
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
lr_model.fit(X_train_scaled, y_train)
svr_model.fit(X_train_scaled, y_train)
mlp_model.fit(X_train_scaled, y_train)

# 7. Generate Predictions for the Test Set
models = {
    "Linear Regression": lr_model.predict(X_test_scaled),
    "Random Forest (Optimized)": rf_model.predict(X_test),
    "SVR": svr_model.predict(X_test_scaled),
    "XGBoost": xgb_model.predict(X_test),
    "Neural Network": mlp_model.predict(X_test_scaled)
}

# 8. Print Metrics Leaderboard
def directional_accuracy(real, pred):
    real_direction = np.sign(real.diff().dropna())
    pred_direction = np.sign(pd.Series(pred, index=real.index).diff().dropna())
    return (real_direction.values == pred_direction.values).mean() * 100

print(f"{'Model':<30} | {'MAE':<10} | {'R2 Score':<10} | {'Direction Acc.':<15}")
print("-" * 75)
for name, preds in models.items():
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    direction_acc = directional_accuracy(y_test, preds)
    print(f"{name:<30} | {mae:<1.4f} | {r2:<1.4f} | {direction_acc:<15.2f}%")

# 9. GRAPHING THE PERFORMANCE (Last 50 days)
plt.figure(figsize=(15, 8))
plt.title(f'{ticker} Actual vs. Predicted Returns (Last 50 Days)')

# Plot Actual Returns
plt.plot(y_test.index[-50:], y_test.values[-50:], label="Actual Returns", color='black', linewidth=2)

# Plot Predictions for key models
plt.plot(y_test.index[-50:], models["Random Forest (Optimized)"][-50:], label="Optimized RF Preds", color='darkgreen', linestyle='--', alpha=0.8)
plt.plot(y_test.index[-50:], models["Neural Network"][-50:], label="NN Preds", color='red', linestyle='--', alpha=0.6)
plt.plot(y_test.index[-50:], models["SVR"][-50:], label="SVR Preds", color='blue', linestyle='--', alpha=0.6)
plt.plot(y_test.index[-50:], models["Linear Regression"][-50:], label="LR Preds", color='yellow', linestyle='--', alpha=0.6)


plt.ylabel("Daily Return (%)")
plt.xlabel("Date")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
