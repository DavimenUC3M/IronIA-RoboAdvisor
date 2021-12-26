import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import random
import datetime
from datetime import date
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt.efficient_frontier import EfficientCVaR,EfficientCDaR
from pypfopt.discrete_allocation import DiscreteAllocation,get_latest_prices
from pypfopt import objective_functions


def CVaR_method(portfolio,risk=0.05,budget=100000,gamma=0.5,verbose=True):
    
    if verbose:
        print("Using pypfopt...")

    mu = mean_historical_return(portfolio)  
    S = portfolio.cov()

    ef_cvar = EfficientCVaR(mu, S,weight_bounds=(0,1),solver = 'CPLEX')   
    
    if gamma != 0:
        ef_cvar.add_objective(objective_functions.L2_reg, gamma=gamma)
        
    cvar_weights = ef_cvar.efficient_risk(target_cvar=risk,market_neutral=False)

    cleaned_weights = ef_cvar.clean_weights()

    latest_prices = get_latest_prices(portfolio)
    ef_cvar.portfolio_performance(verbose=verbose)

    da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=budget)

    return cleaned_weights


def CDaR_method(portfolio,risk=0.05,budget=100000,gamma=0.5,verbose=True):
    
    if verbose:  
        print("Using pypfopt...")
    mu = mean_historical_return(portfolio)  
    S = portfolio.cov()
    ef_cdar = EfficientCDaR(mu, S,weight_bounds=(0,1),solver = 'CPLEX')
    if gamma != 0:
        ef_cdar.add_objective(objective_functions.L2_reg, gamma=gamma) 
    cdar_weights = ef_cdar.efficient_risk(target_cdar=risk,market_neutral=False)

    cleaned_weights = ef_cdar.clean_weights()

    latest_prices = get_latest_prices(portfolio)
    ef_cdar.portfolio_performance(verbose=verbose)

    da_cdar = DiscreteAllocation(cdar_weights, latest_prices, total_portfolio_value=100000)

    return cleaned_weights

def sharpe_method(portfolio,budget=100000,verbose=True):
    
    if verbose:  
        print("Using pypfopt...")

    mu = mean_historical_return(portfolio)  
    S = CovarianceShrinkage(portfolio).ledoit_wolf()
    ef = EfficientFrontier(mu, S,solver = 'CPLEX')
    
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    latest_prices = get_latest_prices(portfolio)
    ef.portfolio_performance(verbose=verbose)

    da_cvar = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=budget)

    return cleaned_weights




