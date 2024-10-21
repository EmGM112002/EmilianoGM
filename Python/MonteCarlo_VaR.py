"""
Value at Risk (VaR)
Value at risk is a statistic measure to predict the maximum lost of a stock market portfolio 
on a specific time horizon given a confidence level.

Definitions
1. Time horizon. Specific period of time in which the risk will be evaluated.
2. Confidence level. Percentage of confidence in which the risk want to be evaluated
3. Expected loss. Loss that will not surpass the specified confidence level.

#!Recommend to have a previously portfolio, we will need the total price and weights later.
"""


#Necessary libraries (install if necessary)
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import datetime as dt 
import yfinance as yf  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from scipy.stats import norm  # type: ignore


#*Set time periods

#endDate = dt.datetime.datetime(2024,12,31) format (Y,M,D)
endDate = dt.datetime.now() #sets date as today

#startDate = dt.datetime.datetime(2018,1,1) #format (Y,M,D)
startDate = endDate - dt.timedelta(365*15)


#*Create portfolio Tickers

tickers = ['SPY','BND','GLD','QQQ','VTI']


#*Download the adjust close prices for the tickers.
#Adjusted closed prices account for dividends and stock splits. The analysis will be more accurate with this method.

adj_close_df = pd.DataFrame() #Storage Tickers data

for ticker in tickers:
  adj_close_df[ticker] = yf.download(ticker, start = startDate, end = endDate)['Adj Close']


#*Calculate return logs and drop NaN's

#Is easier to work with return Log's because they are additive.
log_returns = np.log(adj_close_df / adj_close_df.shift(1)) 
log_returns.dropna(inplace = True)


#*Create the function to calculated Expected returns
#We assume that future returns are based on past returns To be more accurate you should come with 
# your own expected returns or pay for an equity research firms expected returns.

def expected_returns(weights, log_returns):
  return np.sum(log_returns.mean() * weights)


#*Create the covariance matrix

cov_matrix = log_returns.cov()

#*Create a function to calculate portfolio Standard Deviation (std)

def standard_deviation(weights, cov_matrix):
  return np.sqrt(np.dot(weights.T, #transpose the weights colum
                        np.dot(cov_matrix, weights))) #Gets the matrix product of weights and cov_matrix


#*Create the covariance matrix
cov_matrix = log_returns.cov()
cov_matrix


#*Create an equally weighted portfolio and find total portfolio expected return.
portfolio_value = 1 #Total stock prices of the portfolio

weights = np.array([1/len(tickers)]*len(tickers))

portfolio_expected_return = expected_returns(weights, log_returns)

portfolio_std_dev = standard_deviation(weights, cov_matrix)


#*Z-score's function.
#We used this to add a volatile element that is random (gaussian process)

def random_z_score():
  return np.random.normal(0,1)


#*scenario GainLoss function
days = 5

def scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days):
  return portfolio_value * portfolio_std_dev * days + portfolio_value * z_score * np.sqrt(days)


#*Run n simulations

simulations = 1000
scenarioReturns = []

for i in range(simulations):
  z_score = random_z_score()
  scenarioReturns.append(scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days))


#*Specify a confidence interval and calculate VaR
confidence_level = 0.95
VaR = -np.percentile(scenarioReturns , (1 - confidence_level) * 100)
print(VaR)

#*Plot All the scenarios

plt.hist(scenarioReturns , bins = 50 , density = True)
plt.xlabel("Scenario Gain/Loss ($)")
plt.ylabel("Frequency")
plt.title(f"Distribution of Scenario Gain/Loss over {days} days")
plt.axvline(-VaR , color = "r", linestyle = "dashed", linewidth = 2 , label = f"VaR at {confidence_level}% confidence interval")
plt.legend()
plt.show()