import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import tempfile

class BacktestingEMA:
    def __init__(self, ticker, start_date, end_date, initial_investment, ema_period):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.ema_period = ema_period
        self.initial_investment = initial_investment

        # Call other methods
        self.fetch_data()
        self.indicators()
        self.signals()
        self.positions()
        self.returns()

    def fetch_data(self):
        self.df = yf.download(self.ticker, self.start_date, self.end_date)

    def indicators(self):
        self.df['ema'] = self.df['Adj Close'].ewm(span=self.ema_period, min_periods=self.ema_period).mean()

    def signals(self):
        self.df['signal'] = np.where(self.df['Adj Close'] < self.df['ema'], -1, 1)

    def positions(self):
        self.df['positions'] = self.df['signal'].replace(to_replace=0, method='ffill')

    def returns(self):
        self.df['bnh_returns'] = np.log(self.df['Adj Close'] / self.df['Adj Close'].shift(1))
        self.df['strategy_returns'] = self.df['bnh_returns'] * self.df['positions'].shift(1)
        self.df['investment'] = self.initial_investment * self.df['strategy_returns'].cumsum().apply(np.exp)
        self.profit = self.df['investment'][-1] - self.initial_investment
        return self.profit

    def plot_analysis(self):
        # Plot to check the strategy working as planned
        plt.figure(figsize=(15, 6))
        plt.plot(self.df['Adj Close'], label='Price')
        plt.plot(self.df['ema'], label='EMA', linestyle='--')
        plt.title("EMA Strategy - Buy/Sell Signals")
        plt.legend()
        st.pyplot(plt)

        # Generate report using quantstats
        qs.reports.full(self.df['strategy_returns'])

# Streamlit app
st.title('EMA Strategy Backtesting')

# Sidebar
st.sidebar.header('Parameters')
ticker = st.sidebar.text_input('Enter Ticker Symbol', value='INFY')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2023-12-31'))
initial_investment = st.sidebar.number_input('Initial Investment', value=10000)
ema_period = st.sidebar.slider('EMA Period', min_value=1, max_value=50, value=9)

# Instantiate the backtesting class
backtester = BacktestingEMA(ticker, start_date, end_date, initial_investment, ema_period)

# Display analysis
st.header('Analysis')
backtester.plot_analysis()

# Display profit
st.header('Profit')
st.write("Profit on initial investment: $", backtester.profit)

# Display QuantStats HTML report
st.header('QuantStats Report')
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
    qs.reports.html(backtester.df['strategy_returns'], output=tmpfile.name)
    with open(tmpfile.name, 'r') as f:
        report_html = f.read()
    st.components.v1.html(report_html, width=900, height=600, scrolling=True)
