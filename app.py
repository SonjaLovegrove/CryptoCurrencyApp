import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# API Key for News API
NEWS_API_KEY = "72dc824b390d4657b923884464154e54"

# List of cryptocurrencies to download
cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOT-USD", "DOGE-USD", "LTC-USD", "MATIC-USD", "LINK-USD", "XLM-USD", "XMR-USD", "AVAX-USD", "VET-USD", "ALGO-USD", "AAVE-USD", "ATOM-USD", "MANA-USD", "XTZ-USD", "FIL-USD", "EGLD-USD", "ENJ-USD", "EOS-USD", "HBAR-USD", "KSM-USD", "FTT-USD", "QNT-USD", "RUNE-USD", "TRX-USD"]

def download_data():
    data = {}
    for crypto in cryptos:
        df = yf.download(crypto, period="2y")
        df.columns = df.columns.droplevel(1)
        data[crypto] = df

    return data

# Fetch live data and flatten the column index
def fetch_live_data(crypto):
    data = yf.download(crypto, period="2y")
    data.columns = data.columns.droplevel(1)  # Remove the first level of the column index
    return data

# Function to fetch news
def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return articles

# Streamlit app
st.title("Cryptocurrency Analysis and Forecasting")

# Sidebar Navigation
menu = ["Home", "Visualize Data", "Exploratory Data Analysis", "Groups and Correlation", "Forecasting", "Profit Recommendation"]
choice = st.sidebar.selectbox("Navigation", menu)

# Home page
if choice == "Home":
    st.header("Welcome to the Cryptocurrency Analysis and Forecasting Tool")
    st.subheader("Latest Cryptocurrency News")
    news = fetch_news()
    for article in news:
        st.write(article["title"])
        st.write(article["description"])
        st.write(article["url"])

# Page 1: Visualize Data
elif choice == "Visualize Data":
    st.header("Visualize Data")
    crypto = st.selectbox("Select a Cryptocurrency", cryptos)
    data = fetch_live_data(crypto)

    if st.button("Show Moving Average"):
        data["MA_30"] = data["Close"].rolling(window=30).mean()
        st.line_chart(data[["Close", "MA_30"]])

    if st.button("Show Line Chart"):
        st.line_chart(data["Close"])

    if st.button("Show Data Table"):
        st.dataframe(data)

# Page 2: Exploratory Data Analysis
elif choice == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    crypto = st.selectbox("Select a Cryptocurrency for EDA", cryptos)
    data = fetch_live_data(crypto)

    st.write("Summary Statistics")
    st.write(data.describe())

    if st.button("Show Histogram"):
        fig, ax = plt.subplots()
        ax.hist(data["Close"], bins=25)
        st.pyplot(fig)

    if st.button("Show Trends"):
        st.line_chart(data["Close"])

    if st.button("Correlation Matrix"):
        corr = data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True)
        st.pyplot(fig)

    if st.button("Time Series Decomposition"):
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposed = seasonal_decompose(data["Close"], model="multiplicative", period=30)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,10))
        decomposed.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        decomposed.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        decomposed.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        decomposed.resid.plot(ax=ax4)
        ax4.set_title('Residuals')
        plt.tight_layout()
        st.pyplot(fig)

# Page 3: Groups and Correlation Analysis
elif choice == "Groups and Correlation":
    st.header("Groups and Correlation Analysis")

    # Download historical data
    st.write("Downloading historical data...")
    data = download_data()

    # Preprocess data
    combined_data = pd.DataFrame({crypto: data[crypto]["Close"] for crypto in cryptos}).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=4)
    clusters = kmeans.fit_predict(reduced_data)
    combined_data["Cluster"] = clusters

    st.write("Clustering Result:")
    st.dataframe(combined_data)

    st.write("Top Correlations:")
    selected_crypto = st.selectbox("Select a Cryptocurrency", cryptos)

    # Calculate correlations and exclude the selected cryptocurrency
    correlations = combined_data.corr()[selected_crypto].drop(selected_crypto).sort_values(ascending=False)

    # Display top 4 positive and negative correlations
    st.write("Top 4 Positive Correlations:", correlations.head(4))
    st.write("Top 4 Negative Correlations:", correlations.tail(4))

elif choice == "Forecasting":
    st.header("Cryptocurrency Forecasting")
    crypto = st.selectbox("Select a Cryptocurrency to Forecast", cryptos)
    model_choice = st.selectbox("Select a Model", ["ARIMA", "LSTM", "Random Forest", "Prophet"])

    # User-defined forecast horizon
    forecast_steps = st.number_input("Enter the number of days to forecast", min_value=1, max_value=365, value=30)

    # Fetch data
    data = fetch_live_data(crypto)["Close"]

    if model_choice == "ARIMA":
        # Ensure the data is stationary
        def is_stationary(series):
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series)
            p_value = result[1]
            return p_value <= 0.05

        # Apply differencing to make the data stationary
        if not is_stationary(data):
            data = data.diff().dropna()
            if not is_stationary(data):
                data = data.diff().dropna()

        # Fit an ARIMA model
        from statsmodels.tsa.arima.model import ARIMA

        st.write("Fitting ARIMA model...")
        arima_model = ARIMA(data, order=(5, 1, 0))  # Adjust order (p, d, q) as necessary
        arima_fit = arima_model.fit()

        # Forecast the next forecast_steps days
        forecast = arima_fit.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        # Plot actual vs forecast
        fig, ax = plt.subplots()
        ax.plot(data[-100:], label="Actual")
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps,
                                               freq='D')
        ax.plot(forecast_index, forecast_mean, label="Forecast (Mean)", color="orange")
        ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color="orange",
                                alpha=0.2, label="Confidence Interval")
        ax.legend()
        st.pyplot(fig)

        # Evaluate the model
        actual_values = data[-forecast_steps:]
        predicted_values = forecast_mean[:len(actual_values)]

        st.write("Model Evaluation Metrics:")
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)

        st.write(f"RMSE: {rmse}")
        st.write(f"MAE: {mae}")
        st.write(f"MSE: {mse}")
        st.write(f"R²: {r2}")

    elif model_choice == "LSTM":
        st.write("Using LSTM Model...")
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))


        # Prepare data for LSTM
        def create_sequences(data, sequence_length):
            x, y = [], []
            for i in range(len(data) - sequence_length):
                x.append(data[i:i + sequence_length])
                y.append(data[i + sequence_length])
            return np.array(x), np.array(y)


        sequence_length = 60
        x, y = create_sequences(scaled_data, sequence_length)
        train_size = int(len(x) * 0.8)
        x_train, y_train = x[:train_size], y[:train_size]
        x_test, y_test = x[train_size:], y[train_size:]

        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)

        # Predict
        predictions_scaled = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot actual vs predictions
        fig, ax = plt.subplots()
        ax.plot(data[-len(actual_values):].index, actual_values, label="Actual")
        ax.plot(data[-len(actual_values):].index, predictions, label="Forecast (LSTM)", color="orange")
        ax.legend()
        st.pyplot(fig)

        # Calculate evaluation metrics
        st.write("Model Evaluation Metrics:")
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        mae = mean_absolute_error(actual_values, predictions)
        mse = mean_squared_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)

        st.write(f"RMSE: {rmse}")
        st.write(f"MAE: {mae}")
        st.write(f"MSE: {mse}")
        st.write(f"R²: {r2}")

    elif model_choice == "Random Forest":
        st.write("Using Random Forest Model...")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        import numpy as np
        import pandas as pd

        # Ensure data is a DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name="Close")

        # Prepare data with lagged features and moving averages
        data['Lag_1'] = data['Close'].shift(1)
        data['Lag_2'] = data['Close'].shift(2)
        data['Lag_3'] = data['Close'].shift(3)
        data['Lag_4'] = data['Close'].shift(4)
        data['Lag_5'] = data['Close'].shift(5)
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data = data.dropna()  # Drop rows with missing values from lagging

        # Define features and target
        x = data[['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'SMA_5', 'SMA_10']]
        y = data['Close']

        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Train Random Forest with hyperparameter tuning
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(x_train, y_train)

        # Predict future prices using iterative forecasting
        last_5_days = data['Close'].tail(5).values.tolist()
        predictions = []

        for _ in range(forecast_steps):
            next_pred = model.predict([last_5_days + [np.mean(last_5_days[-5:]), np.mean(last_5_days[-10:])]])[0]
            predictions.append(next_pred)
            last_5_days.pop(0)
            last_5_days.append(next_pred)

        # Plot actual vs predictions
        fig, ax = plt.subplots()
        ax.plot(data.index[-100:], data['Close'][-100:], label="Actual")
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
        ax.plot(forecast_index, predictions, label="Forecast (RF)", color="orange")
        ax.legend()
        st.pyplot(fig)

        # Calculate evaluation metrics
        st.write("Model Evaluation Metrics:")
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        actual_values = data['Close'][-forecast_steps:]
        predicted_values = predictions[:len(actual_values)]

        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)

        st.write(f"RMSE: {rmse}")
        st.write(f"MAE: {mae}")
        st.write(f"MSE: {mse}")
        st.write(f"R²: {r2}")

    elif model_choice == "Prophet":
        st.write("Using Facebook Prophet...")
        from prophet import Prophet
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        # Ensure data is a DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name="Close")

        # Prepare data for Prophet
        prophet_data = data.reset_index()
        prophet_data.columns = ['ds', 'y']

        # Fit Prophet model
        model = Prophet()
        model.fit(prophet_data)

        # Predict future values
        future = model.make_future_dataframe(periods=forecast_steps)
        forecast = model.predict(future)

        # Plot forecast
        fig = model.plot(forecast)
        st.pyplot(fig)

        # Extract actual and predicted values for the evaluation period
        actual_values = data['Close'][-forecast_steps:].values
        predicted_values = forecast['yhat'].iloc[-forecast_steps:].values

        # Calculate evaluation metrics
        st.write("Model Evaluation Metrics:")
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)

        # Display the metrics
        st.write(f"RMSE: {rmse}")
        st.write(f"MAE: {mae}")
        st.write(f"MSE: {mse}")
        st.write(f"R²: {r2}")

# Page 5: Profit Recommendation
if choice == "Profit Recommendation":
    st.header("Cryptocurrency Profit Recommendation")

    # User inputs
    profit_target = st.number_input("Enter the profit amount you want to achieve (in USD)", min_value=0.0, step=0.1)
    time_period = st.number_input("Enter the time period (in days) to achieve the profit", min_value=1, max_value=365, value=30)

    st.write("Calculating recommendations...")

    # Import necessary libraries
    from statsmodels.tsa.arima.model import ARIMA
    import numpy as np

    # Fetch live data for all cryptos
    recommended_coins = []
    for coin in cryptos:
        # Fetch data and ensure we're using only the 'Close' prices
        data = fetch_live_data(coin)
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                data = data['Close']
            else:
                st.write(f"Error: 'Close' column not found for {coin}. Skipping...")
                continue

        # Ensure the series has no missing values
        data = data.dropna()

        # Fit the ARIMA model
        try:
            model = ARIMA(data, order=(5, 1, 0)).fit()
        except Exception as e:
            st.write(f"Error fitting ARIMA model for {coin}: {e}")
            continue

        # Forecast future prices
        forecast = model.forecast(steps=time_period)

        # Calculate potential profit
        potential_profit = forecast.iloc[-1] - data.iloc[-1]
        if potential_profit >= profit_target:
            recommended_coins.append({
                "Coin": coin,
                "Current Price": data.iloc[-1],
                "Forecasted Price": forecast.iloc[-1],
                "Potential Profit": potential_profit
            })

    # Display results
    if recommended_coins:
        st.write("Recommended Coins to Achieve Your Profit Target:")
        df_recommendations = pd.DataFrame(recommended_coins)
        st.dataframe(df_recommendations)
    else:
        st.write("No coins meet your profit target within the selected time period.")

