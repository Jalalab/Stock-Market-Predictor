import numpy as np 
import pandas as pd
import yfinance as yf 
from keras.models import load_model 
import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 1. Page layout and title
st.set_page_config(page_title="Stock Marktet", page_icon="📊", layout="wide")

# 2. Custom CSS
st.markdown(
    """
    <style>
    .stButton>button {
        color: white;
        background-color: #FF4B4B;
        border-radius: 10px;
        height: 50px;
        width: 200px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. UI elements
st.title("DASHBOARD")

col1, col2 = st.columns(2)
col1.metric("Revenue", "$10K", "+5%")
col2.metric("Users", "1,200", "-2%")

# Load Bi-LSTM + Attention model
model = load_model(r'C:\Users\Pial\OneDrive\Desktop\pial\python\JN\Untitled Folder\bi_lstm_model.keras')
#streamlit run "C:\Users\Pial\OneDrive\Desktop\pial\python\JN\Untitled Folder\app.py"

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start ='2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock data')
st.write(data)

# Split train-test
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

pass_100_days = data_train.tail(100)
data_test = pd.concat([pass_100_days, data_test], ignore_index=True)

data_test_scaler = scaler.fit_transform(data_test)

# MA Plots
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'b')
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(15,12))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'g')
plt.plot(data.Close, 'b')
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(15,12))
plt.plot(ma_100_days, 'g')
plt.plot(ma_200_days, 'y')
plt.plot(data.Close, 'b')
st.pyplot(fig3)

# Test dataset for prediction
x, y = [], []
for i in range(100, data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

# Inverse scaling
predict = scaler.inverse_transform(predict)
y = scaler.inverse_transform(y.reshape(-1,1))

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'g', label= 'Predicted Price')
plt.plot(y, 'b', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)


# ----------------- FUTURE PREDICTION -----------------
# User chooses horizon
future_days = st.selectbox(
    "Select how many future days to predict:",
    options=[30, 100, 200],
    index=0
)

last_100_days = data_test_scaler[-100:]
future_input = last_100_days.reshape(1, last_100_days.shape[0], 1)

future_predictions = []
for _ in range(future_days):
    pred = model.predict(future_input, verbose=0)   # shape (1,1)
    future_predictions.append(pred[0,0])

    # reshape pred to (1,1,1) before appending
    pred_reshaped = pred.reshape(1,1,1)
    future_input = np.append(future_input[:,1:,:], pred_reshaped, axis=1)

# Proper inverse scaling
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

# Plot future predictions
st.subheader(f'Predicted Price for Next {future_days} Days')
fig5 = plt.figure(figsize=(10,6))
plt.plot(range(len(y)), y, 'b', label="Original Price")
plt.plot(range(len(predict)), predict, 'g', label="Predicted (Test Data)")
plt.plot(range(len(y), len(y)+future_days), future_predictions, 'r', label=f"Next {future_days} Days")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig5)


# ----------------- Accuracy Comparison -----------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from keras.layers import Layer
from tensorflow import keras

# ---------------- CUSTOM ATTENTION LAYER ----------------
#@keras.saving.register_keras_serializable()
class Attention(Layer):
    def __init__(self, return_attention=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.return_attention = return_attention

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        if self.return_attention:
            return [tf.keras.backend.sum(output, axis=1), a]
        return tf.keras.backend.sum(output, axis=1)




# Load both models
lstm_model = load_model(
    r'C:\Users\Pial\OneDrive\Desktop\pial\python\JN\Untitled Folder\LSTM model.keras'
)

bi_lstm_model = load_model(
    r'C:\Users\Pial\OneDrive\Desktop\pial\python\JN\Untitled Folder\bi_lstm_model.h5',
    custom_objects={"Attention": Attention}   # required for custom layer
)

# Predictions from both models
lstm_predict = lstm_model.predict(x)
bi_lstm_predict = bi_lstm_model.predict(x)

# Inverse transform to original scale
lstm_predict = scaler.inverse_transform(lstm_predict)
bi_lstm_predict = scaler.inverse_transform(bi_lstm_predict)
y_true = scaler.inverse_transform(y.reshape(-1,1))

# Accuracy metrics
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

lstm_rmse, lstm_mae, lstm_r2 = get_metrics(y_true, lstm_predict)
bi_rmse, bi_mae, bi_r2 = get_metrics(y_true, bi_lstm_predict)

# Show results in Streamlit
st.subheader("📊 Model Accuracy Comparison")
st.write(pd.DataFrame({
    "Model": ["LSTM", "Bi-LSTM + Attention"],
    "RMSE ↓": [lstm_rmse, bi_rmse],
    "MAE ↓": [lstm_mae, bi_mae],
    "R² Score ↑": [lstm_r2, bi_r2]
}))

# Highlight which model performed better
better_model = "LSTM" if lstm_rmse < bi_rmse else "Bi-LSTM + Attention"
st.success(f"✅ Based on RMSE, the better performing model is: **{better_model}**")

