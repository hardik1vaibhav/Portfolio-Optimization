#!/usr/bin/env python
# coding: utf-8

# # Portfolio Optimization 

# ## Import Required Libraries

# In[4]:


pip install yfinance


# In[5]:


pip install datetime


# In[6]:


pip install timedelta


# In[ ]:


import yfinance as yf 
#This library is used to pick stock prices from website of Yahoo Finance
import pandas as pd
from datetime import datetime, timedelta
#datatime allows us select a certain time range
import numpy as np
from scipy.optimize import minimize
#NumPy and SciPy will allow us to use certain statistical methods that we need


# # Section 1- Define Tickers and Time Range

# ## Define the list of tickers

# In[ ]:


tickers = ['HDB','RELIANCE.NS','TCS.NS','SBIN.NS', 'AXB.IL']


# ## Set the end date to today

# In[ ]:


end_date = datetime.today()
print(end_date)


# ## Set the start date to 5 years ago

# In[22]:


start_date = end_date - timedelta(days = 5*365)
print(start_date)


# # Section 2 - Download Adjusted Closed Prices

# ## Create an empty DataFrame to store the adjusted close prices

# In[23]:


adj_close_df = pd.DataFrame()


# ## Download the close prices for each ticker

# In[24]:


for ticker in tickers:
    data = yf.download(ticker, start = start_date,end = end_date)
    adj_close_df[ticker] = data['Adj Close']


# # Display the DataFrame

# In[25]:


print(adj_close_df)


# # Section 3 - Calculate Lognormal Returns

# ## Calculate the lognormal returns for each ticker

# In[26]:


log_returns = np.log(adj_close_df/adj_close_df.shift(1))


# ## Drop any missing values

# In[27]:


log_returns = log_returns.dropna()
print(log_returns)


# # Section 4 - Calculate Covariance Matrix

# ## Calculate the covariance matrix using annualized returns

# In[28]:


cov_matrix = log_returns.cov()*252
print(cov_matrix)


# # Section 4 - Define Portfolio Performance Metrices

# In[ ]:





# ## Calculate the portfolio standard deviation

# This line of code calculates the portfolio variance, which is a measure of risk associated with a portfolio of assets.  

# In[29]:


def standard_deviation(weights,cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)    


# ## Calculate the expected return

# In[ ]:





# In[30]:


def expected_return(weights,log_returns):
    return np.sum(log_returns.mean()*weights)*252


# ## Calculate the Sharpe Ratio

# In[31]:


def sharpe_ratio(weights,log_returns, cov_matrix, risk_free_rate):
    return(expected_return(weights,log_returns)- risk_free_rate) / standard_deviation(weights,cov_matrix)


# # Section 5 - Portfolio Optimization

# ## Set the Risk-free rate

# In[32]:


risk_free_rate = 0.07 


# ## Define the function to minimize (negative Sharpe Ratio)

# In[34]:


def neg_sharpe_ratio(weights,log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights,log_returns, cov_matrix, risk_free_rate)
    


# ## Set the Constraints and Bounds 

# In[41]:


constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.4) for _ in range(len(tickers))]


# ## Set the Initial Weights 

# In[36]:


initial_weights = np.array([1/len(tickers)]*len(tickers))
print (initial_weights) 


# ##  Optimize the weights to maximize sharpe ratio

# In[38]:


optimized_results = minimize(neg_sharpe_ratio,initial_weights, args =(log_returns, cov_matrix, risk_free_rate), method = 'SLSQP', constraints = constraints, bounds = bounds)


# ## Get the Optimal Weights

# In[39]:


optimal_weights = optimized_results.x


# ## Section 7 - Analyze the optimal Portfolio

# ## Display the analytics of the portfolio

# In[42]:


print('Optimal Weights')
for ticker, weight in zip(tickers,optimal_weights):
    print(f"{ticker}: {weight:.4f}")

optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")


# ## Display the final portfolio as a plot

# In[43]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights)

plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')

plt.show()


# ## Happy Coding

# In[ ]:




