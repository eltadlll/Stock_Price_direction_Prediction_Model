# Stock Market Prediction Project
This repository contains a comprehensive Python script that demonstrates how to download financial data, engineer technical indicators as features, and train multiple machine learning models (XGBoost, Random Forest, Linear Regression, SVR, Neural Network) to predict the next day's stock returns.
The script includes optimizations for handling yfinance data structures in 2026, proper data scaling, and metrics for both prediction error and directional accuracy.
## Project Description
The goal of this project is to compare the performance of various machine learning algorithms in a time-series forecasting context. It moves beyond simple price prediction to focus on daily returns, which is a more statistically sound approach for financial modeling.
### Key features:
  Data acquisition using yfinance.
  Feature engineering with over a dozen pandas-ta-classic indicators.
  Training of 5 different regression models.
  Evaluation using MAE, R², and Directional Accuracy.
  Visualization of performance.
### How to Run the Code
Save the script: Copy the complete code from the last response and save it as a Python file (e.g., predict_stocks.py).
Execute: Run the script from your terminal:
bash
python predict_stocks.py
Use code with caution.

Review Output: The console will display a performance leaderboard, and a graph will pop up showing the actual vs. predicted returns for the last 50 days.
Code Overview
The script is broken down into modular sections:
Configuration
we can easily change the ticker symbol (AAPL) and the start date:
python
ticker = 'AAPL' 
date = '2010-01-01'
length2 = [14, 20, 50]
Use code with caution.

## Key Logic

#### df.ta.macd(append=True): Uses the strategy system to quickly generate technical features.

#### df['Target'] = df['Close'].pct_change().shift(-1): This crucial step changes the prediction target from raw price to percentage return.

####StandardScaler: Applied to all models that are sensitive to data range (NN, SVR, Linear Regression).

### Interpreting the Results
The output metrics table is the most important part: Model	MAE (Mean Abs. Error)	R2 Score	Direction Acc.
The highest is linear regression with values 	0.0123 MAE	-0.1520 R2	58.20% Directional Accuracy
MAE: The average percentage point error.
R² Score: A low score (e.g., near 0.0) is normal for returns prediction and means the model captures a tiny edge. A negative score means the model is useless.
Directional Accuracy: Anything above 51% is a potential edge in trading.
