### FRAMEWORKS AND DEPENDENCIES
import copy
#from google.cloud import bigquery
import os
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
from PIL import Image, ImageFilter
from collections import OrderedDict
import matplotlib as mpl
import streamlit as st

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
import os

import pandas_bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
pandas_bokeh.output_notebook()


from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt.efficient_frontier import EfficientCVaR,EfficientCDaR
from pypfopt.discrete_allocation import DiscreteAllocation,get_latest_prices
from pypfopt import objective_functions

from pyomo.environ import *
from pyomo.opt import SolverFactory
# print('mira puto')
# print(os.getcwd())
from src.test_pipeline import test_pipeline
from src.test_pipeline import test_rolling
from src.test_pipeline import random_test
from src.test_pipeline import Hierarchical_Computing

plt.rcParams["figure.figsize"] = (18,5)



# LOAD DATA

complete_df = pd.read_csv("data/complete_df.csv")
betas = pd.read_csv("data/betas.csv")
category = pd.read_csv('data/category.csv')


train = pd.read_csv("data/prices_train.csv")
train.set_index("Unnamed: 0",inplace=True)
train.index.name= 'date'
test = pd.read_csv("data/prices_test.csv")
test.set_index("Unnamed: 0",inplace=True)
test.index.name= 'date'

with open('data/different_funds_10.pkl', 'rb') as f: #Cleaning duplicated name funds
    DifferentNameFunds = pickle.load(f)
    
train = train[DifferentNameFunds]
test = test[DifferentNameFunds]

train_risk_1 = pd.read_csv("data/Risk Dataframes/train_1_risk.csv")
train_risk_1.set_index("date",inplace=True)
train_risk_1 = train_risk_1.reindex(columns=DifferentNameFunds)
train_risk_1.dropna(axis=1,inplace=True)

train_risk_2 = pd.read_csv("data/Risk Dataframes/train_2_risk.csv")
train_risk_2.set_index("date",inplace=True)
train_risk_2 = train_risk_2.reindex(columns=DifferentNameFunds)
train_risk_2.dropna(axis=1,inplace=True)

train_risk_3 = pd.read_csv("data/Risk Dataframes/train_3_risk.csv")
train_risk_3 = train_risk_3.reindex(columns=DifferentNameFunds)
train_risk_3.dropna(axis=1,inplace=True)

train_risk_4 = pd.read_csv("data/Risk Dataframes/train_4_risk.csv")
train_risk_4.set_index("date",inplace=True)
train_risk_4 = train_risk_4.reindex(columns=DifferentNameFunds)
train_risk_4.dropna(axis=1,inplace=True)

train_risk_5 = pd.read_csv("data/Risk Dataframes/train_5_risk.csv")
train_risk_5.set_index("date",inplace=True)
train_risk_5 = train_risk_5.reindex(columns=DifferentNameFunds)
train_risk_5.dropna(axis=1,inplace=True)



### Title 
def header():
    
        st.markdown("<h1 style='text-align: center;'>IronIA's Roboadvisor</h1>",unsafe_allow_html=True)
        ### Description
        st.markdown("""<p style='text-align: center;'>This is a pocket application focused on advising individuals on starting investing
        on the financial world. This app is for those who have the basic ideas of how they want to invert, but don't have 
        enough knowledge to make their own investment portfolio.</p>""",unsafe_allow_html=True)

        ### Image
        st.image("data/robo-advisor.png")

### Function for performing an Apply on the choosen funds (extracting extra data from the category csv)
def add_extra_info(bench_id):
    sub_filter = category[category['benchmark_finametrix_id'] == bench_id]
    indx = sub_filter.index[0]
    return category.iloc[indx][['benchmark','category','morningstar_category_id']]
### Controllers 
def controllers2():
    # Select the anomaly to detect 
    st.sidebar.markdown("<h1 style='text-align: center;'>User Demands</h1>",unsafe_allow_html=True)
    option_risk = st.sidebar.selectbox('Select a risk Measure',['CVaR', 'CDaR', 'MAD','ML','sharpe'],help=""" 
            - Conditional Value at Risk (CVaR) :  Risk assessment measure that quantifies the amount of tail risk an investment portfolio has.
            - Conditional Drawdown at Risk (CDaR) : Risk measure which quantifies in aggregated format the number and magnitude of the portfolio drawdowns over some period of time.
            - MaxLoss (ML) 
            - Mean Absolute Deviation (MAD)
            - Sharpe Ratio (sharpe) : Average return earned in excess of the risk-free rate per unit of volatility or total risk.""")
    # Filtering anomalies
    st.sidebar.markdown('''
                    <h4 style='text-align: center;'>Choose the following Measures: </h4>
                    
                    - Risk Level        : Percentageo of Risk the Client is willing to accept
                    - Budget           : Amount of money the Client is willing to invest                
                    ''',unsafe_allow_html=True)

    risk_lvl      = st.sidebar.slider(label="Risk Level",min_value=0.0,max_value=1.0,value=0.2,step=0.005,help="Between 0 and 1, choose a value. Keep in mind that the lower the value you choose the lower risk you are taking and thus you are being more conservative. ")
    budget      = st.sidebar.number_input('Insert your Investment Budget',min_value=0,value=2000,help="Total amount of money the you want to expend in this Portfolio" )
    return option_risk,risk_lvl,budget

    


def main():

            sys.path.insert(0,"..")
            ### Title
            st.set_page_config(layout="wide")
            header()
            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>User Portfolio</h3>",unsafe_allow_html=True)
            
            # Reuse the Controllers output
            option_risk,risk_lvl,budget = controllers2()


            weights,returns2,info_dict = test_pipeline(train,test,market_neutral=False,samples=500,min_weight=0.05,add_leftovers=False,method=option_risk,risk=risk_lvl,budget=budget,gamma=0.1,rs=40) #Methods = CDaR, CVaR, sharpe, MAD, ML
            
            # In order to show aditional info of the choosen funds
            funds_inversion = [i * budget for i in list(weights.values())]
            total_investment = sum(funds_inversion)
            remaining_budget = budget- total_investment
            choosen_funds = list(returns2.columns[:-1])
            choosen_funds_info = complete_df.loc[complete_df['names'].isin(choosen_funds)]
            choosen_funds_info['budget inversion'] = funds_inversion
            choosen_funds_info[['benchmark','category','morningstar_category_id']] = choosen_funds_info.benchmark_id.apply(add_extra_info)
            choosen_funds_info = choosen_funds_info[['fund_indx','benchmark_id','names','budget inversion','risk_level','category','benchmark','morningstar_category_id']]
        

            p = returns2.plot_bokeh.line(
                figsize=(900, 500),
                title="Evolution of budget",
                xlabel="Date",
                ylabel="Your budget [$]",
                panning=False,
                zooming=False,
                legend="top_left")
            st.bokeh_chart(p)


            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>General Results on Test Year</h3>",unsafe_allow_html=True)

            # Setting commas 
            def place_value(number):
                return ("{:,}".format(number))
            
            # print(place_value(1000000))

            col1, col2, col3,col4,col5 = st.columns(5)
            col1.metric("Invested Budget", place_value(round(total_investment,2))+'$')
            col2.metric("Remaining Budget", place_value(round(remaining_budget,2))+'$')
            col3.metric("Volatility",str(info_dict['test_volatility']))
            col4.metric("Total Returns", str(info_dict['test_return'])+ '%' )
            col5.metric("Money Obtained", place_value(info_dict['money_test_year'])+'$')

            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>Aditional Fund's Information</h3>",unsafe_allow_html=True)
            st.dataframe(choosen_funds_info)
          

if __name__=="__main__":
    main()