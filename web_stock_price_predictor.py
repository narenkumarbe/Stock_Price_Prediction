import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler



st.title('Stock Price Prediction App')
ticker_name = st.text_input('Enter Stock Ticker', 'GOOG')

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

stock_data = yf.download(ticker_name, start, end)

model = load_model('Latest_stock_price_model.keras')

st.subheader('Stock data')
st.write(stock_data)

splitting_len = int(len(stock_data)*0.7)
x_test = pd.DataFrame(stock_data.Close)[splitting_len:]

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_data, 'r')    
    return fig 

st.subheader('Original Close Price and MA for 250 days')
stock_data['MA_250'] = stock_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), stock_data['MA_250'], stock_data, 0))

st.subheader('Original Close Price and MA for 200 days')
stock_data['MA_200'] = stock_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), stock_data['MA_200'], stock_data, 0))

st.subheader('Original Close Price and MA for 100 days')
stock_data['MA_100'] = stock_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), stock_data['MA_100'], stock_data, 0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), stock_data['MA_100'], stock_data, 1, stock_data['MA_250']))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = stock_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data.tail(10))

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))

plt.plot(pd.concat([stock_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)


