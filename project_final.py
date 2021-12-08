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

# # Output Locations
# TICKERS_LOC = "B3"
# WEIGHTS_LOC = "C3"
# AVG_RETURN_LOC = "D3"
# STATUS_LOC = "H2"
# STATUS_DEC_LOC = "I2"
# TRACK_STATUS_LOC = "H2"
# TRACK_STATUS_DEC_LOC = "I2"
# TIME_LOC = "H3"
# TRACK_TIME_LOC = "H3"
# P_VAR_LOC = "H11"
# P_AVG_RETURN_LOC = "H12"
# P_ASSET_COUNT = "J11"
# REALIZED_RETURN_LOC = "C11"

# # Input Locations
# INTERVAL_LOC = "N3"
# MODE_LOC = "N5"
# PERIOD_LOC = "N7"
# START_LOC = "N9"
# END_LOC = "N10"
# TRACK_INTERVAL_LOC = "D3"
# TRACK_START_LOC = "D5"
# TRACK_END_LOC = "D6"


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


def track_portfolio(tickers, weights, wb):
    # tickers is a yf.Tickers object     weights is an array of the weights     wb is an xlwings book caller object
    
    # get settings from wb
    interval = wb.sheets['Tracking'].range(TRACK_INTERVAL_LOC).value
    start = wb.sheets['Tracking'].range(TRACK_START_LOC).value
    end = wb.sheets['Tracking'].range(TRACK_END_LOC).value + dt.timedelta(days=1)

    # download and modify ticker history
    history = tickers.history(interval=interval, start=start, end=end, threads=True, auto_adjust=True)
    data = history["Close"].dropna(axis=0,how='all')

    # calculate total returns of each asset
    returns = data.iloc[-1]/data.iloc[0]

    # calculate weighted return of portfolio 
    weighted_realized_returns = returns*weights
    realized_return = sum(weighted_realized_returns) - 1

    # output return to wb 
    wb.sheets['Tracking'].range(REALIZED_RETURN_LOC).value = realized_return

    # PLOT RETURNS
    # Get returns for each interval
    periodic_returns = data / data.iloc[0]

    # weight the returns
    weighted_periodic_returns = periodic_returns * weights

    # sum to get portfolio return
    periodic_P_return = weighted_periodic_returns.sum(axis=1) - 1

    # pandas to matplotlib conversion
    pd.plotting.register_matplotlib_converters()

    # generate figure object
    fig = plt.figure()
    # fig.suptitle("Portfolio Return")

    # generate plot of returns
    ax = plt.subplot()
    ax.set_title("Portfolio Return")

    indexList = periodic_P_return.index.tolist()
    xaxis = [x.strftime("%Y-%m-%d") for i,x in enumerate(indexList) if i % int(len(indexList)/5) == 0]

    ax.plot(periodic_P_return)
    ax.set_xticks(xaxis)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.grid(True)

    # output plot to wb
    wb.sheets['Plot'].pictures.add(fig, name="Test", update=True)





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

    # TODO update row to contain ratios
    # portfolioRow['Sharpe Ratio'] = 0.212

    return portfolioRow
    # # return for track_portfolio()
    # return tickers, weights, wb








# def main():
# 	try:
# 		# get risk parity portfolio and update wb
# 		tickers, weights, wb = riskparity()
# 		# track this portfolio and update wb
# 		track_portfolio(tickers, weights, wb)

# 	# explain EailedDownload error on wb
# 	except FailedDownload as ex:
# 		wb = xw.Book.caller()
# 		wb.sheets['Input'].range(STATUS_LOC).value = "ERROR"
# 		wb.sheets['Input'].range(STATUS_DEC_LOC).value = "Failed to download " + str(ex.msg)
# 		raise 
	
# 	# explain other error on wb
# 	except Exception as ex:
# 		wb = xw.Book.caller()
# 		wb.sheets['Input'].range(STATUS_LOC).value = "ERROR"
# 		wb.sheets['Input'].range(STATUS_DEC_LOC).value = str(ex)
# 		raise 






def main():
    # define the Excel book
    wb = xw.Book.caller()

    # get tickers
    tickList = wb.sheets['Main'].range("Assets").expand('down').value

    # generate tickers (list(yfinance.Ticker))
    tickers = yf.Tickers(tickList)

    # get table
    table = wb.sheets['Main'].range('MainTable[[#all]]').options(pd.DataFrame, index=False).value
    table = table.dropna(subset=['Portfolio Type', 'Sample Start', 'Sample End', 'Tracking Start', 'Tracking End'])
    print(table)

    # download start
    donwloadStart = table['Sample Start'].min()
    print(donwloadStart)

    # download end
    downloadEnd = table['Tracking End'].max() + dt.timedelta(days=1)
    print(downloadEnd)

    # get interval
    interval = wb.sheets['Main'].range("interval").value
    print(interval)

    # download stock data
    history = tickers.history(interval=interval, start=donwloadStart, end=downloadEnd, threads=True, auto_adjust=True)
    closeHistory = history['Close']
    print(closeHistory)

    # check that all tickers are valid else rase FailedDownload error
    if (list(yf.shared._ERRORS.keys())):
        ex = FailedDownload(list(yf.shared._ERRORS.keys()))
        raise ex

    for index, row in table.iterrows():
        print(row) 

        # relaventData = closeHistory.loc[row['Sample Start']: row['Tracking End']]
        # print(relaventData)

        # run that type of portfolio analysis 
        if (row["Portfolio Type"] == "Risk Parity"):
            table.loc[index] = riskparity(row, closeHistory, tickList)

    print(table)








if __name__ == "__main__":
    xw.Book("project_final.xlsm").set_mock_caller()
    main()

