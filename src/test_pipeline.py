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

from src.risk_methods_pyomo import pyomo
from src.risk_methods_pyportfolio import *

def random_test(train,test,iterations,n_splits,print_every):
    
    random_gains = []
    
    for i in range(0,iterations):
        
        if ((i+1) % print_every == 0):
            print(i+1,"out of",iterations)
            
        N_ = n_splits
        train_random = train.sample(n=N_,axis=1)
        test_ = test.pct_change().dropna()

        def rd(n, total_sum):
            nums = np.random.rand(n)
            return nums/np.sum(nums)*total_sum

        random_weights = rd(N_,1)

        random_weights_dict = {}

        for i,j in enumerate(train_random.columns):
            random_weights_dict[j] = random_weights[i]

        choosen_funds_df = test_[random_weights_dict.keys()].copy()
        # for i,j in enumerate(random_weights_dict.keys()):
        #     choosen_funds_df[j] = choosen_funds_df[j] * list(random_weights_dict.values())[i] 
        #     plt.plot(np.cumsum(choosen_funds_df[j]),label=j)

        # plt.legend(loc='upper left')
        # plt.title("Evolution of funds by weights")
        pass

        test_returns = test_rolling(random_weights_dict,test_,verbose=False)
        random_gains.append(round(test_returns[-1]*100,2))
        
    return random_gains

def test_rolling(weights_dict,test,budget=1,verbose=True):
    choosen_funds_df = test[weights_dict.keys()].copy()

    for i,j in enumerate(weights_dict.keys()):
        choosen_funds_df[j] = choosen_funds_df[j] * list(weights_dict.values())[i] 
        #plt.plot(np.cumsum(choosen_funds_df[j]),label=j)

    weights_dict = dict(sorted(weights_dict.items(), key=lambda item: item[1],reverse=True))
    keys = list(weights_dict.keys())[0:10]
    returns = np.cumsum(choosen_funds_df.sum(axis=1)) 
    returns2 = (np.cumsum(choosen_funds_df.loc[:,keys])+1)*budget
    returns = (returns+1)*budget
    returns_ = returns.to_frame()
    returns_ = returns_.rename(columns={0:'Total'})
    returns2['Total'] = returns_['Total']
    
    #plt.plot(choosen_funds_df)
  
    returns = np.cumsum(choosen_funds_df.sum(axis=1))

    if verbose:
        print("Volatility obtained in test:",np.std(choosen_funds_df.sum(axis=1)*np.sqrt(len(test)+1)))                
        print(f"Total return obtained in the test year: {round(returns[-1]*100,2)}%")
        print(f"Money obtained during test year: {round(returns[-1]*budget,2)}$")
        print(f"Distributed in {len(weights_dict)} funds:")
        print(weights_dict) 

        returns2.plot_bokeh.line(
            figsize=(900, 500),
            title="Evolution of budget",
            xlabel="Date",
            ylabel="Your budget [$]",
            panning=False,
            zooming=False,
            legend="top_left")

    return returns 

def test_pipeline(train,test,samples=0,min_weight=0,add_leftovers=True,method="CVaR",market_neutral=False,risk=0.05,budget=100000,gamma=0.5,rs=40,verbose=True):
    
    if samples !=0:
        train_ = train.sample(n=samples,axis=1,random_state=rs)
    else:
        train_ = train.copy()
    test_ = test.pct_change().dropna()
    
    if verbose:
        print(f"Using {method} method")

    opt_time_start = time.time()
    if method == "CVaR":
        if market_neutral==True:
            cleaned_weights = pyomo(train_,"CVaR",risk=risk,verbose=verbose)
        else:
            cleaned_weights = CVaR_method(train_,risk=risk,budget=budget,gamma=gamma,verbose=verbose) 
    elif method == "CDaR":
        if market_neutral==True:
            cleaned_weights = pyomo(train_,"CDaR",risk=risk,verbose=verbose)
        else:
            cleaned_weights = CDaR_method(train_,risk=risk,budget=budget,gamma=gamma,verbose=verbose)  
    elif method == "sharpe":
        cleaned_weights = sharpe_method(train_,budget=budget,verbose=verbose)  
    
    elif method == "MAD":
        cleaned_weights = pyomo(train_,"MAD",risk=risk,verbose=verbose)
        
    else:
        cleaned_weights = pyomo(train_,"ML",risk=risk,verbose=verbose)
    opt_time_end = time.time()
    
    if verbose:
        print(f"Optimization time: {opt_time_end-opt_time_start} seconds")

    weights_dict = {}
    discarded_wights = {}
    for key in cleaned_weights.keys():
        if cleaned_weights[key] > min_weight:
            weights_dict[key] = cleaned_weights[key]
        
        elif cleaned_weights[key] > 0 and cleaned_weights[key] <= min_weight:
            discarded_wights[key] = cleaned_weights[key]
    
    if add_leftovers:
    
        leftovers = sum(discarded_wights.values())/len(weights_dict)

        for key in weights_dict.keys():
            weights_dict[key] += leftovers
    
    if verbose:
        print(f"Budget not inverted: {round(budget-sum(weights_dict.values())*budget,2)}$")
    
    returns = test_rolling(weights_dict,test_,budget,verbose)
    
    
    return weights_dict  


def Hierarchical_Computing(train,test,n_steps=2,min_weight=0,add_leftovers=True,split_size=100,print_every=50,
                           market_neutral=False,method="CVaR",budget=100,risk=0.005,gamma=0.2,verbose=True):
    
    for step in range(0,n_steps):
          
        if step == 0:    
            
            if verbose:
                print("Iteration 1")
                print("-----------")
            
            train_all_splits = []
            n_splits = int(np.ceil(len(train.columns)/split_size))

            for i in range(0,n_splits): 
                train_all_splits.append(train.iloc[:,i*split_size:(i+1)*split_size])

            selected_funds = []
            for counter,train_ in enumerate(train_all_splits):
                
                if verbose and (counter+1)%print_every == 0:
                    print(f"{counter+1} out of {n_splits}")
                try:
                    weights = test_pipeline(train_,test,market_neutral=False,method=method,
                                            min_weight=min_weight,add_leftovers=True,risk=risk,
                                            budget=budget,gamma=gamma,verbose=False)
                except:
                    print("Error at position:",counter, "solving with gamma=0 for this partition")
                    weights = test_pipeline(train_,test,market_neutral=False,method=method,
                                            min_weight=min_weight,add_leftovers=True,risk=risk
                                            ,budget=budget,gamma=0,verbose=False)

                for i in weights.keys():
                    selected_funds.append(i)

            if verbose:
                print(f"Selected funds in step 1: {len(selected_funds)} \n")
        
        else:
            
            if verbose:
                print(f"Iteration {step+1}")
                print("-----------")
            
            train_all_splits = []
            n_splits = int(np.ceil(len(train[selected_funds].columns)/split_size))

            for i in range(0,n_splits): 
                train_all_splits.append(train[selected_funds].iloc[:,i*split_size:(i+1)*split_size])

            selected_funds = []
            for counter,train_ in enumerate(train_all_splits):
                
                if verbose and (counter+1)%(print_every//(step+1)) == 0:
                    print(f"{counter+1} out of {n_splits}")
                    
                try:
                    weights = test_pipeline(train_,test,market_neutral=False,method=method,
                                            min_weight=min_weight,add_leftovers=True,risk=risk,
                                            budget=budget,gamma=gamma,verbose=False)
                except:
                    print("Error at position:",counter)
                    weights = test_pipeline(train_,test,market_neutral=False,method=method,
                                            min_weight=min_weight,add_leftovers=True,risk=risk,
                                            budget=budget,gamma=0,verbose=False)

                for i in weights.keys():
                    selected_funds.append(i) 

            if verbose:
                print(f"Selected funds in step {step+1}: {len(selected_funds)} \n")

    return selected_funds