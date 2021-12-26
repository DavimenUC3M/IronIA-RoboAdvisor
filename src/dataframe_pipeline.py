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

from google.cloud import bigquery
import os     
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="data/ironia-data-dfc6335a0abe.json"

client = bigquery.Client()


def generate_df(ids,working_dates,period=1,risk=0.05):

  pd.options.mode.chained_assignment = None

  main_df = pd.DataFrame(columns=["Ironia_id","Name","MDD","DaR","CDaR","RF","VaR", "CVaR", "MAD"])

  prices_df = pd.DataFrame()

  for i in ids:
    sql = f'SELECT nav,date,name FROM `ironia-data.Ironia_Ext.precios` WHERE Ironia_id={i[0]} and date BETWEEN "2016-01-01" and "2018-12-31" LIMIT 10000 '
    df = pd.read_gbq(sql)
    df.dropna(inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by="date",inplace=True)
    df.set_index(df["date"],inplace=True)

    if len(df) >= len(working_dates): #Number of working days from 2016 to 2018

      try:
        df = df.loc[working_dates]
        df["nav"] = df["nav"].interpolate(method="linear")
        full_series = True
      
      except:
        full_series = False
      
      if full_series:
        prices_df = pd.merge(prices_df,df["nav"],how="outer", left_index=True, right_index=True)
        prices_df.rename(columns={'nav':df["name"].values[0] }, inplace=True)

        prices_df = prices_df.interpolate(method="linear")


        last_date = date.fromisoformat(str(df[-1:].index.values)[2:12])
        years_ago = (last_date - relativedelta(years=period)).isoformat()
  
        df = df[years_ago:last_date]

        df.drop("date",axis=1,inplace=True)
        df["nav return"] = (df["nav"]-df["nav"].shift(1))/df["nav"].shift(1)
        df["nav return"][0] = 0
        VaR = np.percentile(df["nav return"],risk*100) 
        CVaR = np.mean(df["nav return"].loc[df["nav return"]<=VaR])
        max_acum = df["nav"].cummax()
        df["drawdown"] = -(df["nav"]-max_acum)/max_acum
        MDD =  np.max(df["drawdown"].values)
        recovery_factor = df["nav"].iloc[-1]/np.max(df["drawdown"].values)
        DaR = np.percentile(df["drawdown"],risk*100)
        CDaR = np.mean(df["drawdown"].loc[df["drawdown"]<=DaR])
        MAD = df["nav return"].mad()
        main_df = main_df.append({"Ironia_id":int(i[0]),"Name":df["name"].values[0],"MDD":MDD,"DaR":DaR,"CDaR":CDaR,"RF":recovery_factor,"VaR":VaR, "CVaR":CVaR, "MAD":MAD}, ignore_index=True)
      
  return [main_df,prices_df]

def get_train_test_prices(ids,working_dates_all,start="2016-01-01",train_span="2018-12-31",test_span="2019-12-31",delta=0,print_every=100):

  pd.options.mode.chained_assignment = None

  index = pd.date_range(start=start,end=test_span)
  prices_df = pd.DataFrame(index=index,columns=["Init_"])

  for count,i in enumerate(ids):
    sql = f'SELECT nav,date,name FROM `ironia-data.Ironia_Ext.precios` WHERE Ironia_id={i[0]} and date BETWEEN "{start}" and "{test_span}" LIMIT 10000 '
    df = pd.read_gbq(sql)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by="date",inplace=True)
    df.set_index(df["date"],inplace=True)

    if ((count+1) % print_every) == 0:
      print(count+1,"out of",len(ids))

    if len(df) >= (len(working_dates_all) - delta) : #Number of working days from 2016 to 2019, the delta is to admit series with more NaNs
      
      prices_df = pd.merge(prices_df,df["nav"],how="outer", left_index=True, right_index=True)
      prices_df.rename(columns={'nav':df["name"].values[0] }, inplace=True)
      prices_df = prices_df.loc[working_dates_all]

      prices_df = prices_df.interpolate(method="linear")
      
  test_start = date.fromisoformat(train_span)
  test_start = (test_start + relativedelta(days=1)).isoformat()
  prices_df.drop("Init_",inplace=True,axis=1)
  prices_df.dropna(axis='columns',inplace=True)

  return [prices_df[start:train_span],prices_df[test_start:test_span]]  