# based on: https://quantdare.com/risk-parity-in-python/
import pandas as pd
import xlwings as xw
import yfinance as yf
from scipy.optimize import minimize
import numpy as np
import math
import time
import datetime as dt
import matplotlib.pyplot as plt
TOLERANCE = 0.000000000000001


# exception for failed yfinance download 
class FailedDownload(Exception):
	'''Raised when ticker(s) is invalid'''
	def __init__(self, msg):
		self.msg = msg

# exception for failed covariance matrix
class FailedCov(Exception):
	'''Raised when ticker(s) is invalid'''
	def __init__(self, msg):
		self.msg = msg





# NOT MY WORK ->


def _allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

    # It returns the risk of the weights distribution
    return portfolio_risk


def _assets_risk_contribution_to_allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) \
        / portfolio_risk

    # It returns the contribution of each asset to the risk of the weights
    # distribution
    return assets_risk_contribution


def _risk_budget_objective_error(weights, args):

    # The covariance matrix occupies the first position in the variable
    covariances = args[0]

    # The desired contribution of each asset to the portfolio risk occupies the
    # second position
    assets_risk_budget = args[1]

    # We convert the weights to a matrix
    weights = np.matrix(weights)

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = \
        _assets_risk_contribution_to_allocation_risk(weights, covariances)

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = \
        np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))

    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = \
        sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]

    # It returns the calculated error
    return error


def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):

    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})

    # Optimisation process in scipy
    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=None,
                               options={'disp': True, 'ftol': TOLERANCE})

    # Recover the weights from the optimised object
    weights = optimize_result.x

    # It returns the optimised weights
    return weights






# MY WORK ->



def riskparity(portfolioRow, history, tickList):
    
    sampleData = history.loc[portfolioRow['Sample Start']: portfolioRow['Sample End']]

    returns = sampleData.pct_change()
    av_returns = returns.mean()

    # get the covariance matrix
    try:
    	cov = returns.cov()
    except Exception as ex:
    	raise FailedCov('Failed to create covariance matrix') from ex

    # determine how much risk should be associated with each asset
    assets_risk_budget = [1 / len(tickList)] * len(tickList)

    # initial weights (guess)
    init_weights = [1 / len(tickList)] * len(tickList)

    # get risk parity weights
    covList = cov.values.tolist()
    weights = _get_risk_parity_weights(covList, assets_risk_budget, init_weights)

    # create a weighted covariance matrix and total variance
    WeightedCov = covList
    p_variance = 0
    for r, row in enumerate(WeightedCov):
        for c, col in enumerate(row):
            WeightedCov[r][c] *= weights[r]*weights[c]
            p_variance += WeightedCov[r][c]
    WeightedCov = pd.DataFrame(WeightedCov, cov.index, cov.columns)
    p_return = sum(av_returns*weights)

    colList= history.columns.tolist()
    weightsDF = pd.DataFrame(columns=colList, index=[0])
    for i, col in enumerate(colList):
    	weightsDF[col][0] = weights[i]

    return weightsDF




def portfolioHistory(data, weights):
	returns = data.pct_change()
	returns = returns.iloc[1:]
	stkList = weights.columns.tolist()
	for stk in stkList:
		returns[stk] *= weights[stk][0]
	pChanges = returns.sum(axis=1)
	return pChanges



def portfolioReturn(data, weights):
	# return = final price / initial price
	stkList = weights.columns.tolist()
	pReturn = 0 
	for stk in stkList:
		pReturn += (data[stk].iloc[-1] / data[stk].iloc[0]) * weights[stk][0]
	return pReturn - 1 



def sharpeRatio(pChanges):
	dailyExpectedReturn = pChanges.mean()
	dailyStdev = pChanges.std()
	annualizedExpectedReturn = (1+dailyExpectedReturn)**252 - 1
	annualizedStdev =  dailyStdev * (252**(1/2))
	return annualizedExpectedReturn / annualizedStdev



def trackPortfolio(portfolioRow, history, weights):
	trackData = history.loc[portfolioRow['Tracking Start']: portfolioRow['Tracking End']]

	pChanges = portfolioHistory(trackData, weights)

	pReturn = portfolioReturn(trackData, weights) 
	portfolioRow["Return"] = pReturn

	pSharpe = sharpeRatio(pChanges)
	portfolioRow["Sharpe Ratio"] = pSharpe




def main():
    # define the Excel book
    wb = xw.Book.caller()

    s = time.time()

    # get tickers
    tickList = wb.sheets['Main'].range("Assets").expand('down').value

    # generate tickers (list(yfinance.Ticker))
    tickers = yf.Tickers(tickList)

    # get table
    table = wb.sheets['Main'].range('MainTable[[#all]]').options(pd.DataFrame, index=False).value
    table = table.dropna(subset=['Portfolio Type', 'Sample Start', 'Sample End', 'Tracking Start', 'Tracking End'])

    # download start
    donwloadStart = table['Sample Start'].min()

    # download end
    downloadEnd = table['Tracking End'].max() + dt.timedelta(days=1)

    # get interval
    interval = wb.sheets['Main'].range("Interval").value

    # download stock data
    history = tickers.history(interval=interval, start=donwloadStart, end=downloadEnd, threads=True, auto_adjust=True)
    closeHistory = history['Close']

    # check that all tickers are valid else rase FailedDownload error
    if (list(yf.shared._ERRORS.keys())):
        ex = FailedDownload(list(yf.shared._ERRORS.keys()))
        raise ex

    for index, row in table.iterrows():
        # run that type of portfolio analysis 
        if (row["Portfolio Type"] == "Risk Parity"):
            portfolioWeights = riskparity(row, closeHistory, tickList)
            trackPortfolio(row, closeHistory, portfolioWeights)
            table.loc[index] = row 
            # print(table)


    wb.sheets['Main'].tables('MainTable').update(table, index=False)

    wb.sheets['Main'].range('Status').value = "COMPLETE"
    wb.sheets['Main'].range('Time').value = time.time() - s







if __name__ == "__main__":
    xw.Book("project_final.xlsm").set_mock_caller()
    main()

