import numpy as np
import yfinance as yf
from datetime import date, datetime
from plotly import graph_objs as go
import streamlit as st

st.title('Fast Fourier Transform on Financial Time Series')

st.markdown('FFT is used on a financial time series to extract the signal from the noise. The time series is transform the frequencies where a threshold can be defined. This can act as a filter. The signal is then transformed back using the inverse FFT.')

START = '2020-01-01'
TODAY = date.today().strftime('%Y-%m-%d')
SYMBOLS = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'GOOGL', 'FB', 'TSM',
           'TSLA', 'BABA', 'V', 'MA', 'BAC', 'PYPL', 'NVDA', 'ADBE']


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


def plot_data(x, y, series_name, title, slider_visible=False):
    fig = go.Figure()
    fig.add_trace(go.Line(x=x, y=y, name=series_name))
    fig.layout.update(title_text=title, xaxis_rangeslider_visible=slider_visible)
    st.plotly_chart(fig)


def normalize(data):
    dt = 1/data.shape[0]
    t = np.arange(0, 1, dt)
    f = data['Adj Close']
    return t, f, dt

def compute_fft(t, f, dt):
  n = len(t)
  f_hat = np.fft.fft(f, n)
  power_spectrum_density = f_hat * np.conj(f_hat) / n
  frequency = (1 / (dt * n)) * np.arange(n)
  return frequency, power_spectrum_density, f_hat

def reverse_fft(power_spectrum_density, f_hat, threshold):
  indices = power_spectrum_density > threshold
  power_spectrum_density_clean = power_spectrum_density * indices
  f_hat = indices * f_hat
  f_filtered = np.fft.ifft(f_hat)
  return power_spectrum_density_clean, f_filtered

stocks = (SYMBOLS)
selected_stock = st.selectbox(
    'Ticker Symbol',
    stocks
)

st.date_input('Start date', value=datetime.strptime(START, '%Y-%m-%d'))


data = load_data(selected_stock)

st.subheader('Raw Data')
st.write(data.tail())


st.subheader('Graphs')

plot_data(x=data['Date'], y=data['Adj Close'], series_name='Price', title='Time Series', slider_visible=True)

t, f, dt = normalize(data)
x, psd, f_hat = compute_fft(t, f, dt)


L = np.arange(1, np.floor(len(t) / 2), dtype='int')
plot_data(x=x[L], y=abs(psd[L]), series_name='Frequency', title='Power Spectrum Density', slider_visible=True)

st.subheader('Filter')
threshold = st.number_input(label='Threshold', value=200, step=1)

psd_clean, f_filtered = reverse_fft(psd, f_hat, threshold)

st.subheader('Output')


fig = go.Figure()
fig.add_trace(go.Line(x=data['Date'], y=data['Adj Close'], name='Raw'))
fig.add_trace(go.Line(x=data['Date'], y=abs(f_filtered), name='Filtered'))
fig.layout.update(title_text=selected_stock + ' (Raw vs Filtered)', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)


fig = go.Figure()
fig.add_trace(go.Line(x=x, y=abs(psd[L]), name='Raw'))
fig.add_trace(go.Line(x=x, y=abs(psd_clean[L]), name='Filtered'))
fig.layout.update(title_text='Power Spectrum Density (Raw vs Filtered)', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)
