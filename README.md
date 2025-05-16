# 🪙 CryptoCurrency Forecasting App

An interactive cryptocurrency forecasting platform built using Streamlit, offering multi-model predictions (ARIMA, LSTM, Random Forest, Prophet) for key cryptocurrencies. The tool supports exploratory analysis, price prediction, and visualization of crypto trends using live or historical data.

## 🚀 Project Overview

This project was developed as part of a portfolio initiative to apply a range of machine learning and time series forecasting techniques to real-world financial data. It enables users to:

- Choose between multiple cryptocurrencies (e.g. BTC, ETH)
- Select a prediction model (ARIMA, LSTM, Prophet, Random Forest)
- Visualize historical trends and future forecasts
- Access insights through an intuitive web-based interface

## 🔧 Features

- 📈 **ARIMA**: Traditional statistical model for time series forecasting
- 🔮 **LSTM**: Deep learning model well-suited for sequential data
- 🌲 **Random Forest**: Ensemble regression using historical lagged features
- 🧙 **Prophet**: Facebook’s modular forecasting model for seasonality/trend
- 📊 **Interactive visualizations** with Streamlit and Plotly
- 🔌 **Modular architecture** for easily swapping in new models

## 📁 Repository Structure

```
CryptoCurrencyApp/
├── app.py                    # Main Streamlit interface
├── models/                   # Forecasting models (ARIMA, LSTM, etc.)
│   ├── arima.py
│   ├── lstm.py
│   ├── prophet_model.py
│   └── random_forest.py
├── utils/                    # Data processing and API utils
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** for front-end UI
- **Scikit-learn** for ML models
- **TensorFlow / Keras** for LSTM
- **Statsmodels** for ARIMA
- **Facebook Prophet**
- **Plotly & Matplotlib** for charts

## 📦 Installation & Running Locally

Clone the repository:

```bash
git clone https://github.com/SonjaLovegrove/CryptoCurrencyApp.git
cd CryptoCurrencyApp
```

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## 🙋‍♀️ Author

Built by [Sonja Lovegrove](https://github.com/SonjaLovegrove)  
Feel free to connect or contribute!
