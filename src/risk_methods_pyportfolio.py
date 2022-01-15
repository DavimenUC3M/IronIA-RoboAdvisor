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


def obtain_betas_risklevel(portfolio):
    # Fund names
    fund_names = list(portfolio.columns)

    # Obtaining their respective betas
    mapping_df = pd.read_csv('data/complete_df.csv')
    betas = pd.read_csv('data/betas.csv')
    filtered_mappping = mapping_df[mapping_df.names.isin(fund_names)] 
    ids = list(filtered_mappping['fund_indx'])
    filtered_betas = betas[betas.allfunds_id.isin(ids)]
    filtered_betas['value_product'] = filtered_betas['value_product'].apply(lambda x: round(x,3))
    train_betas = filtered_betas[filtered_betas['period'] == '5y']
    train_betas = train_betas[['allfunds_id','value_product']]
    median_betas = train_betas.groupby('allfunds_id').median()
    betas = np.array(median_betas['value_product'])

    ## Obtaining their respective risk Level 
    filtered_mappping = mapping_df[mapping_df.names.isin(fund_names)] 
    risk_lvls = np.array(filtered_mappping['risk_level'])

    return betas, risk_lvls

def CVaR_method(portfolio,risk=0.05,risk_level=0,budget=5000,gamma=0.2,market_neutral=False,verbose=True,solver='CPLEX'):
    
    if verbose:
        print("Using pypfopt...")

    betas, risk_lvls = obtain_betas_risklevel(portfolio)

    mu = mean_historical_return(portfolio)  
    S = portfolio.cov()

    ef_cvar = EfficientCVaR(mu, S,weight_bounds=(0,1),solver = solver)   
    
    if gamma != 0:
        ef_cvar.add_objective(objective_functions.L2_reg, gamma=gamma)

    if risk_level == 1:
        ## New Objective: Minimize Risk Level
        ef_cvar.add_constraint(lambda x: x@risk_lvls <= 2)
        ef_cvar.add_constraint(lambda x: x@risk_lvls >= 1)

    elif risk_level == 2:
        ## New Objective: Minimize Risk Level
        ef_cvar.add_constraint(lambda x: x@risk_lvls <= 3)
        ef_cvar.add_constraint(lambda x: x@risk_lvls >= 2)

    elif risk_level == 3:
        ## New Objective: Minimize Risk Level
        ef_cvar.add_constraint(lambda x: x@risk_lvls <= 4)
        ef_cvar.add_constraint(lambda x: x@risk_lvls >= 3)

    elif risk_level >= 4:
        ## New Objective: Minimize Risk Level
        ef_cvar.add_constraint(lambda x: x@risk_lvls <= 5)
        ef_cvar.add_constraint(lambda x: x@risk_lvls >= 4)
        

    if market_neutral:
        k = 0.01
        ef_cvar.add_constraint(lambda x:x@betas <= k)
        ef_cvar.add_constraint(lambda x:x@betas >= -k)

        
    cvar_weights = ef_cvar.efficient_risk(target_cvar=risk,market_neutral=False)

    cleaned_weights = ef_cvar.clean_weights()

    latest_prices = get_latest_prices(portfolio)
    ef_cvar.portfolio_performance(verbose=verbose)

    da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=budget)

    return cleaned_weights


def CDaR_method(portfolio,risk=0.05,risk_level=0,budget=5000,gamma=0.2,market_neutral=False,verbose=True,solver='CPLEX'):
    
    if verbose:  
        print("Using pypfopt...")

    betas, risk_lvls = obtain_betas_risklevel(portfolio)

    mu = mean_historical_return(portfolio)  
    S = portfolio.cov()

    ef_cdar = EfficientCDaR(mu, S,weight_bounds=(0,1),solver = solver)

    if gamma != 0:
        ef_cdar.add_objective(objective_functions.L2_reg, gamma=gamma) 

    if risk_level == 1:
        ## New Objective: Minimize Risk Level
        ef_cdar.add_constraint(lambda x: x@risk_lvls <= 2)
        ef_cdar.add_constraint(lambda x: x@risk_lvls >= 1)

    elif risk_level == 2:
        ## New Objective: Minimize Risk Level
        ef_cdar.add_constraint(lambda x: x@risk_lvls <= 3)
        ef_cdar.add_constraint(lambda x: x@risk_lvls >= 2)

    elif risk_level == 3:
        ## New Objective: Minimize Risk Level
        ef_cdar.add_constraint(lambda x: x@risk_lvls <= 4)
        ef_cdar.add_constraint(lambda x: x@risk_lvls >= 3)

    elif risk_level >= 4:
        ## New Objective: Minimize Risk Level
        ef_cdar.add_constraint(lambda x: x@risk_lvls <= 5)
        ef_cdar.add_constraint(lambda x: x@risk_lvls >= 4)
        

    if market_neutral:
        k = 0.01
        ef_cdar.add_constraint(lambda x:x@betas <= k)
        ef_cdar.add_constraint(lambda x:x@betas >= -k)

    cdar_weights = ef_cdar.efficient_risk(target_cdar=risk,market_neutral=False)

    cleaned_weights = ef_cdar.clean_weights()

    latest_prices = get_latest_prices(portfolio)
    ef_cdar.portfolio_performance(verbose=verbose)

    da_cdar = DiscreteAllocation(cdar_weights, latest_prices, total_portfolio_value=budget)

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




