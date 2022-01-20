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
from matplotlib import cm
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

import plotly.graph_objects as go

from src.test_pipeline import test_pipeline
from src.test_pipeline import test_rolling
from src.test_pipeline import random_test
from src.test_pipeline import Hierarchical_Computing

plt.rcParams["figure.figsize"] = (18,5)



@st.cache
def data_loader():
        print('Loading ... ')
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

        with open('data/different_funds_7.pkl', 'rb') as f: #Cleaning duplicated name funds
            DifferentNameFunds = pickle.load(f)
            
        train = train[DifferentNameFunds]
        test = test[DifferentNameFunds]

        return complete_df,betas,category,train,test



### Title 
def header():

            html_header="""

            <head>
            <title>PControlDB</title>
            <meta charset="utf-8">
            <meta name="keywords" content="IroAdvisor , Your Personal Fund Manager">
            <meta name="description" content="IroAdvisor Your Personal Fund Manage">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            </head>
            <h1 style="font-size:300%; color:#008080; font-family:Georgia"> IROADVISOR <br>
            <h2 style="color:#008080; font-family:Georgia"> Your Personal Fund Manager</h3> <br>
            <hr style= "  display: block;
            margin-top: 0.5em;
            margin-bottom: 0.5em;
            margin-left: auto;
            margin-right: auto;
            border-style: inset;
            border-width: 1.5px;"></h1>
        """
            st.set_page_config(
                page_title = "IroAdvisor — Your Personal Fund Manager",
                page_icon = Image.open('./data/crop_circle.png')   , 
                layout = "wide",
                initial_sidebar_state = "auto")
            st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
            st.markdown(html_header, unsafe_allow_html=True)

            # st.markdown(""" <style>
            # #MainMenu {visibility: hidden;}
            # footer {visibility: hidden;}
            # </style> """, unsafe_allow_html=True)
            html_header1="""

                <h2 style="font-size:300%; color:#008080; font-family:Georgia">Risk Level Questionnaire</h2>
            """

            st.markdown(html_header1, unsafe_allow_html=True)
            with st.expander('If you have low financial knowledge, we recommend you to fill this Questionnaire'):
                
                

                html_header1="""
                    <h5 style="color:#008080; text-align:center;font-family:Georgia"> We will ask you 7 questions with the aim 
                    of getting to know you better <br> and in this way discard certain funds.</h3> <br>
                """
                st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)                
                st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
                
                st.markdown(html_header1, unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
                score = 0

                col1, col2,col3,col4,col5= st.columns([3,10,5,10,3])
                with col1:
                    st.write("")
    
                with col2:
                    q1 = st.radio(
                    "1. If you had to choose between more job security with a small pay increase and less job security with a big pay increase, which would you pick?",
                    ('A. Definitely more job security with a small pay increase', 
                    'B. Probably more job security with a small pay increase', 
                    'C. Probably less job security with a big pay increase',
                    'D. Definitely less job security with a big pay increase '))
                    if 'A.' in  q1:
                        score += 4
                    elif 'B.' in  q1:
                        score += 3
                    elif 'C.' in  q1:
                        score += 2
                    elif 'D.' in  q1:
                        score += 1   
                with col3:
                    st.write("") 
                with col4:
                    q4 = st.radio(
                    "4. Which of the statements better reflect the way you feel in situations in which you have little to no control over the outcome?",
                    ('A. I tend to panic and start making bad decisions.', 
                    'B. I feel powerless and start overthinking.', 
                    'C. I get a bit nervous but I let the situation develop.',
                    'D. I remain completely calm.'))
                    if 'A.' in  q4:
                        score += 4
                    elif 'B.' in  q4:
                        score += 3
                    elif 'C.' in  q4:
                        score += 2
                    elif 'D.' in  q4:
                        score += 1   
                with col5:
                    st.write("")   


                st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
                col1, col2,col3,col4,col5= st.columns([3,10,5,10,3])
                
                with col1:
                    st.write("")
    
                with col2:
                    q2 = st.radio(
                    "2. Imagine you were in a job where you could choose to be paid salary, commission, or a mix of both. Which would you pick?",
                    ('A. All salary', 
                    'B. Mainly salary ', 
                    'C. Mainly commission ',
                    'D. All commission'))
                    if 'A.' in  q2:
                        score += 4
                    elif 'B.' in  q2:
                        score += 3
                    elif 'C.' in  q2:
                        score += 2
                    elif 'D.' in  q2:
                        score += 1   
                with col3:
                    st.write("") 
                with col4:
                    q5 = st.radio(
                    "5. Of the following investments, which of the following scenarios would you be most comfortable with:",
                    ('A. You can lose down to -2%, and gain up to +9%', 
                    'B. You can lose down to -7%, and gain up to +13%', 
                    'C. You can lose down to -15%, and gain up to +26%',
                    'D. You can lose down to -31%, and gain up to +48%'))
                    if 'A.' in  q5:
                        score += 4
                    elif 'B.' in  q5:
                        score += 3
                    elif 'C.' in  q5:
                        score += 2
                    elif 'D.' in  q5:
                        score += 1  
                with col5:
                    st.write("")   


                st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
                col1, col2,col3,col4,col5= st.columns([3,10,5,10,3])
                
                with col1:
                    st.write("")
    
                with col2:
                    st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
                    q3 = st.radio(
                    "3. When investing, you are primarily concerned about:",
                    ('A. Not losing money (combat inflation)', 
                    'B. Keeping the money you invest and making a bit more', 
                    'C. Relatively consistent growth over time',
                    'D. Making as much money as possible from your investments'))
                    if 'A.' in  q3:
                        score += 4
                    elif 'B.' in  q3:
                        score += 3
                    elif 'C.' in  q3:
                        score += 2
                    elif 'D.' in  q3:
                        score += 1  
                with col3:
                    st.write("") 
                with col4:
                    q6 = st.radio(
                    "6. Back in 2008, the market took a major hit and stocks went down nearly 30%. If you had owned stocks at that time, how would you have reacted (or your real reaction if you actually did have money invested).",
                    ('A. You prefer losing some money than risk losing any more: sell everything!', 
                    'B. Just to be safe, you prefer to sell some of your assets and keep a small part.', 
                    'C. Do nothing! Let the market flow and see how it plays.',
                    'D. Drawdown? Buy more, now that the price is low!'))
                    if 'A.' in  q6:
                        score += 4
                    elif 'B.' in  q6:
                        score += 3
                    elif 'C.' in  q6:
                        score += 2
                    elif 'D.' in  q6:
                        score += 1   
                with col5:
                    st.write("")   


                st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
                
                 
                col1, col2,col3 = st.columns([12,10,7])
                
                with col1:
                    st.write("")
                with col2:
                    q7 = st.radio(
                    "7. Your current age is:",
                    ('A. Over 50', 
                    'B. Between 35 and 49', 
                    'C. Between 25 and 34',
                    'D. Under 25'))
                    if 'A.' in  q7:
                        score += 4
                    elif 'B.' in  q7:
                        score += 3
                    elif 'C.' in  q7:
                        score += 2
                    elif 'D.' in  q7:
                        score += 1   
                with col3:
                    st.write("")
 

                st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
                agree = st.checkbox('Have you finished filling the Questionnaire?',help='If you want to modify several answers (after having clicked this checkbox), we recommend you to unclick the checkbox so that the program does not recalculate when you are modifying the Questionnaire.')
        
                if agree:
                    if score in range(21,29):
                       
                        lower_limit= 1     
                    elif score in range(15,21):
                        
                        lower_limit= 2   
                    elif score in range(8,15):
                        
                        lower_limit= 3    
                            
                    elif score in range(1,8):
                        
                        lower_limit= 4
                else:
                    
                    lower_limit= 0

            html_header1="""

                    <hr style= "  display: block;
                    margin-top: 0.5em;
                    margin-bottom: 0.5em;
                    margin-left: auto;
                    margin-right: auto;
                    border-style: inset;
                    border-width: 1.5px;"></h4>
                """
            st.markdown(html_header1, unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)


            return lower_limit
 
                
            

        
def initial_metrics(info_dict,budget):
            html_card_header1="""
            <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #eef9ea; padding-top: 5px; width: 300px;
            height: 50px;">
                <h3 class="card-title" style="background-color:#eef9ea; color:#008080; font-family:Georgia; text-align: center; padding: 0px 0;">Volatility</h3>
            </div>
            </div>
            """

            html_card_header2="""
            <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #eef9ea; padding-top: 5px; width: 300px;
            height: 50px;">
                <h3 class="card-title" style="background-color:#eef9ea; color:#008080; font-family:Georgia; text-align: center; padding: 0px 0;">Total Returns</h3>
            </div>
            </div>
            """
            html_card_header3="""
            <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #eef9ea; padding-top: 5px; width: 300px;
            height: 50px;">
                <h3 class="card-title" style="background-color:#eef9ea; color:#008080; font-family:Georgia; text-align: center; padding: 0px 0;">Money Obtained</h3>
            </div>
            </div>
            """
            html_card_header4="""
            <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #eef9ea; padding-top: 5px; width: 300px;
            height: 50px;">
                <h3 class="card-title" style="background-color:#eef9ea; color:#008080; font-family:Georgia; text-align: center; padding: 0px 0;">Total Money</h3>
            </div>
            </div>
            """
            
            with st.container():
                col1, col2, col3, col4, col5, col6, col7, col8,col9 = st.columns([1,15,1,15,1,15,1,15,1])
                with col1:
                    st.write("")
                with col2:
                    st.markdown(html_card_header1, unsafe_allow_html=True)
                    fig_c1 = go.Figure(go.Indicator(
                        mode="number",
                        value=info_dict['test_volatility'],
                        number={'suffix': "%", "font": {"size": 40, 'color': "#008080", 'family': "Arial"}, 'valueformat': '.3f'}))
                    fig_c1.update_layout(autosize=False,
                                        width=350, height=90, margin=dict(l=20, r=20, b=20, t=30),
                                        paper_bgcolor="#fbfff0", font={'size': 20})
                    st.plotly_chart(fig_c1)
                    
                with col3:
                    st.write("")
                    
            
                with col4:
                    st.markdown(html_card_header2, unsafe_allow_html=True)
                    fig_c2 = go.Figure(go.Indicator(
                        mode="number",
                        value= info_dict['test_return'],
                        number={'suffix': "%", "font": {"size": 40, 'color': "#008080", 'family': "Arial"}, 'valueformat': '.2f'}))
                    fig_c2.update_layout(autosize=False,
                                        width=350, height=90, margin=dict(l=20, r=20, b=20, t=30),
                                        paper_bgcolor="#fbfff0", font={'size': 20})

                    st.plotly_chart(fig_c2)
                    
                with col5:
                    st.write("")
                with col6:
                    st.markdown(html_card_header3, unsafe_allow_html=True)
                    fig_c3 = go.Figure(go.Indicator(
                        mode="number",
                        value= info_dict['money_test_year'],
                        number={'suffix': "$","font": {"size": 40, 'color': "#008080", 'family': "Arial"}, 'valueformat': '.2f'}))
                    fig_c3.update_layout(autosize=False,
                                        width=350, height=90, margin=dict(l=20, r=20, b=20, t=30),
                                        paper_bgcolor="#fbfff0", font={'size': 20})

                    st.plotly_chart(fig_c3)
                    
                with col7:
                    st.write("")
                with col8:
                    st.markdown(html_card_header4, unsafe_allow_html=True)
                    fig_c4 = go.Figure(go.Indicator(
                        mode="number",
                        value= budget+info_dict['money_test_year'],
                        number={'suffix': "$", "font": {"size": 40, 'color': "#008080", 'family': "Arial"}, 'valueformat': '.2f'}))
                    fig_c4.update_layout(autosize=False,
                                        width=350, height=90, margin=dict(l=20, r=20, b=20, t=30),
                                        paper_bgcolor="#fbfff0", font={'size': 20})

                    st.plotly_chart(fig_c4)
                with col9:
                    st.write("")                                    
            html_br="""
            <br>
            """
            st.markdown(html_br, unsafe_allow_html=True)


def user_portfolio(weights,returns2):

            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            with st.container():
                col1, col2, col3 = st.columns([12,0.5,9])
                with col1:
                        p = returns2.plot_bokeh.line(
                        figsize=(650, 500),
                        title="Evolution of budget",
                        xlabel="Date",
                        ylabel="Your budget [$]",
                        panning=False,
                        zooming=True,
                        legend="top_left")
                        p.legend.label_text_font_size = '8pt'
                        st.bokeh_chart(p)
                with col2:
                    st.write("")
                with col3:
                    
                    st.markdown("<h3 style='text-align: center;color:#008080; font-family:Georgia;'>Your Investments</h3>",unsafe_allow_html=True)
                    #PIE
                    f_names= []
                    data = []
                    for elem in weights:
                        if weights[elem]> 0:
                            data.append(weights[elem])
                            f_names.append(elem)
                    # fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
                    def func(pct):
                        return "{:.1f}%".format(pct)
                    
                    fig = go.Figure(data=[go.Pie(labels=f_names, values=data, hole=.3,sort=True)])
                    # fig.update_layout(  title_text="Your Investments")
                    st.plotly_chart(fig)  


### Computations
@st.cache
def perform_test_pipe(option_risk,budget,train, test,lower_limit):
    
    selected_funds = Hierarchical_Computing(train,test,market_neutral=False,n_steps=2,split_size=500,print_every=20,
                                        min_weight=0.001,add_leftovers=False,method=option_risk,risk_level=lower_limit,risk=0.05,gamma=0.2)

    weights,returns2,info_dict = test_pipeline(train[selected_funds],test,market_neutral=False,
                        min_weight=0.04,add_leftovers=True,samples=0,method=option_risk,risk_level=lower_limit,
                        risk=0.05,budget=budget,gamma=0.15,rs=40) #Methods = CDaR, CVaR, sharpe, MAD, ML

    return weights,returns2,info_dict

### Function for performing an Apply on the choosen funds (extracting extra data from the category csv)
def add_extra_info(bench_id,category):
    sub_filter = category[category['benchmark_finametrix_id'] == bench_id]
    indx = sub_filter.index[0]
    return category.iloc[indx][['benchmark','category','morningstar_category_id']]



### Controllers 
def controllers2():
    
    ### Description
    # st.sidebar.markdown("""<p style='text-align: center;'>This is a pocket application focused on advising individuals on starting investing
    #     on the financial world. This app is for those who have the basic ideas of how they want to invert, but don't have 
    #     enough knowledge to make their own investment portfolio.</p>""",unsafe_allow_html=True)
    st.sidebar.image("data/complete_logo.png")

    st.sidebar.markdown("<h1 style='text-align: center;'>Choose the following Measures</h1>",unsafe_allow_html=True)

    option_risk = st.sidebar.selectbox('Select a Risk Measure',['CVaR', 'CDaR', 'MAD','ML','sharpe'],help=""" 
                WARNING --> ¡Leave CVaR if you are not used to these terms!

                - Conditional Value at Risk (CVaR) :  Risk assessment measure that quantifies the amount of tail risk an investment portfolio has.
                - Conditional Drawdown at Risk (CDaR) : Risk measure which quantifies in aggregated format the number and magnitude of the portfolio drawdowns over some period of time.
                - MaxLoss (ML) 
                - Mean Absolute Deviation (MAD)
                - Sharpe Ratio (sharpe) : Average return earned in excess of the risk-free rate per unit of volatility or total risk.""")

      
    # risk_lvl = st.sidebar.slider(label="Risk Level",min_value=0.0,max_value=1.0,value=0.2,step=0.005,help="Between 0 and 1, choose a value. Keep in mind that the lower the value you choose the lower risk you are taking and thus you are being more conservative. ")
    budget  = st.sidebar.number_input('Insert your Investment Budget',min_value=0,value=2000,help="Total amount of money the you want to expend in this Portfolio" )
    
    st.sidebar.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
    st.sidebar.markdown('''
                    
                    - <h3>Risk Evaluation Method </h3> Method that the Algorithm will use in order to perform the Risk Optimization (we recommend to use CVaR or CDaR).
                    - <h3>Budget        </h3> Amount of money the Client is willing to invest.              
                    ''',unsafe_allow_html=True)

    st.sidebar.markdown("<h1 style='text-align: center;'></h1>",unsafe_allow_html=True)
    st.sidebar.markdown("<h1 style='text-align: center;'>PARTNERED WITH</h1>",unsafe_allow_html=True)
    st.sidebar.markdown("<h1 style='text-align: center;'></h1>",unsafe_allow_html=True)
    st.sidebar.image("data/uc3m_logo.png")
    st.sidebar.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
    st.sidebar.image("data/aliance.png")
    st.sidebar.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
    st.sidebar.image("data/ironia.png")
    
    return option_risk,budget


# Setting commas 
def place_value(number):
    return ("{:,}".format(number))

def main():
            
            sys.path.insert(0,"..")
    ### PAGE TITLE + ICON
            lower_limit=header()
            
    #### LOAD THE DATA AND PERFORM THE OPERATIONS 
            complete_df,betas,category,train,test=data_loader()
            
            # Reuse the Controllers output
            option_risk,budget = controllers2()
            
            # Call our Function for performing all the computations
            weights,returns2,info_dict = perform_test_pipe(option_risk,budget,train,test,lower_limit)


            
            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)

            html_subtitle="""
            <h2 style="color:#008080; font-family:Georgia;"> User Portfolio </h2>
            """
            st.markdown(html_subtitle, unsafe_allow_html=True)      

            ### Portfolio Evolution Chart + Pie Chart ##########################################################################
            user_portfolio(weights,returns2)

            ### Summary Metrics of the Portfolio ##########################################################################
            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            initial_metrics(info_dict,budget)
        
            
            ### Additional Funds Information ##########################################################################
            #COMPUTATIONS
            # In order to show additional info of the choosen funds
            funds_inversion = [str(round(i*budget,3))+'$' for i in list(weights.values())]   # [str(round(i * 100,2))+'%' for i in list(weights.values())]
            choosen_funds = list(returns2.columns[:-1])
            choosen_funds_info = complete_df.loc[complete_df['names'].isin(choosen_funds)]
            choosen_funds_info['budget inversion'] = funds_inversion
            choosen_funds_info[['benchmark','category','morningstar_category_id']] = choosen_funds_info.benchmark_id.apply(lambda x: add_extra_info(x,category))
            choosen_funds_info = choosen_funds_info[['names','benchmark_id','budget inversion','risk_level','category','benchmark','morningstar_category_id']]


            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            html_subtitle="""
            <h2 style="color:#008080; font-family:Georgia;"> Additional Fund's Information</h2>
            """
            st.markdown(html_subtitle, unsafe_allow_html=True) 
            # st.dataframe(choosen_funds_info)

            st.markdown("<h3 style='text-align: center;'></h3>",unsafe_allow_html=True)
            table = "<table>\n"


            # Create the table's column headers
            head = ['Names','Benchmark Id','Budget Inversion','Risk Level','Category','Benchmark','Morningstar Category Id']
            table += '  <tr style="background-color:#eef9ea; color:#008080; font-family:Georgia; font-size: 15px">\n'
            for column in head:
                table += "    <th>{0}</th>\n".format(column.strip())
                
            table += "  </tr>\n"

            # # Create the table's row data
            for line in choosen_funds_info.to_numpy().tolist():
                row =line 
                table += "  <tr>\n"
                col_count=0
                for column in row:
                    table += "    <td>{0}</td>\n".format(str(column).strip())
                    col_count+=1
                table += "  </tr>\n"

            table += "</table>"
            st.markdown(table, unsafe_allow_html=True)



if __name__=="__main__":
    main()

 #textColor="#989595"   