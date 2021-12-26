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
import pickle

from pyomo.environ import *
from pyomo.opt import SolverFactory

def pyomo(portfolio,rsk_metric,risk=0.05,verbose=True): # rsk_metric = CVaR, CDaR, MAD, ML
      
    if verbose:    
        print("Using pyomo...")
    
    prices_df_main = portfolio.copy()
    N = len(prices_df_main.columns)

    betas = np.random.normal(0, 0.01, N)

    with open('working_dates.txt') as f:
        contents = f.read()

    dates_ = contents.split("\n")[1:-1] #Working dates for train

    J = len(dates_)
    rate_return_df = prices_df_main.pct_change().dropna()
    mean_daily_returns = np.array(rate_return_df.mean()) # C

    model = ConcreteModel()

    model.Weights = RangeSet(0,N-1)
    model.Y = RangeSet(0,J-1)
    model.x = Var(model.Weights, domain=NonNegativeReals,bounds=(0,1))

    def obj_expression(model): 
        return sum(model.x[i]*mean_daily_returns[i] for i in model.Weights)

    model.OBJ = Objective(rule=obj_expression, sense=maximize)

    def sum_one(model): 
        return sum(model.x[i] for i in model.Weights) <= 1
    model.cons1 = Constraint(rule=sum_one)

    W = risk

    if rsk_metric == 'MAD':
        #W = 0.005
        model.u_plus = Var(model.Y,domain=NonNegativeReals,bounds=(0.0,None))
        model.u_minus = Var(model.Y,domain=NonNegativeReals,bounds=(0.0,None))

        def dummy_cons(model):
            return  (1/J)*sum(model.u_plus[j] + model.u_minus[j] for  j in model.Y) <= W

        model.cons1_MAD = Constraint(rule = dummy_cons) 

        summatory = sum(sum(rate_return_df.loc[dates_[n]][p] *model.x[p] for p in model.Weights) for n in model.Y)
        def c5_MAD(model,j):
            date = dates_[j]
            return sum(rate_return_df.loc[date][i] *model.x[i] for i in model.Weights) - ((1/J)*summatory) == (model.u_plus[j] - model.u_minus[j] )
        model.cons2_MAD = Constraint(model.Y,rule = c5_MAD) 

    if rsk_metric == 'CVaR':
        #W = 0.005
        ### CVaR 
        model.slack = Var()
        alpha = 1-risk # Confidence level 
        model.w = Var(model.Y,domain=NonNegativeReals)

        def dummy_cons(model):
            return model.slack + (1/(1-alpha))*(1/J)*sum(model.w[j] for j in model.Y) <= W

        model.cons1_CVaR = Constraint(rule = dummy_cons) 

        def c5_CVaR(model,j):
            date = dates_[j]
            return -sum(rate_return_df.loc[date][i] *model.x[i] for i in model.Weights )- model.slack <= model.w[j]
        model.cons2_CVaR = Constraint(model.Y,rule = c5_CVaR) 

    if rsk_metric == 'CDaR':
        #W = 0.1
        ### CDaR 
        model.slack = Var()
        model.z = Var()
        alpha = 1-risk  # Confidence level 
        model.w = Var(model.Y,domain=NonNegativeReals)

        def dummy_cons(model):
            return model.slack + (1/(1-alpha))*(1/J)*sum(model.w[j] for j in model.Y) <= W

        model.cons1_CDaR = Constraint(rule = dummy_cons) 

        def c5_CDaR(model,j):
            act_sum = rate_return_df.loc[:dates_[j]].sum()
            return  model.z - sum( act_sum[i]* model.x[i] for i in model.Weights)- model.slack <= model.w[j]
        model.cons2_CDaR = Constraint(model.Y,rule = c5_CDaR) 

        def c5_max_(model,j):
            act_sum = rate_return_df.loc[:dates_[j]].sum()
            return  sum(act_sum[i]* model.x[i] for i in model.Weights) <= model.z 
        model.cons3_CDaR = Constraint(model.Y,rule = c5_max_) 

    if rsk_metric == 'ML':
        ### MAX LOSS
        model.w = Var(RangeSet(0,0),domain=NonNegativeReals,bounds=(0,1))

        def dummy_cons(model):
            return model.w[0] <= W
        model.cons1_ML = Constraint(rule = dummy_cons) 

        def c5_MaxLoss(model,j):
            date = dates_[j]
            return -sum(rate_return_df.loc[date][i] *model.x[i] for i in model.Weights ) <= model.w[0]
        model.cons2_ML = Constraint(model.Y,rule = c5_MaxLoss)  

    K = 0.001
    def market_neutral_upper(model): 
        return sum(model.x[i]*betas[i] for i in model.Weights) <= K
    model.cons2 = Constraint(rule=market_neutral_upper)

    def market_neutral_lower(model): 
        return sum(model.x[i]*betas[i] for i in model.Weights) >= -K

    model.cons3 = Constraint(rule=market_neutral_lower)

    Results = SolverFactory('cplex').solve(model)

    weights_dict = {}

    for i in model.x:     
        if model.x[i].value > 0:
            weights_dict[prices_df_main.columns[i]] = model.x[i].value

    return weights_dict