import pandas as pd
import numpy as np
import statistics


def volume_rm(df):
    #df = pd.read_csv(path)
    df.drop('Volume', axis=1, inplace=True)
    #df.to_csv(path)
    return df

def base_currency(df):
    #df = pd.read_csv(path)
    df.rename(columns={'Currency': 'Fx_Currency'}, inplace=True)
    df2 = df.assign(Base_Currency='USD')
    #df2.to_csv(path)
    return df2

def avg_ochl(df):
    #df = pd.read_csv(path)
    avg_values = []
    for idx, row in df.iterrows():
        avg = round(((row['High']+row['Low'])/2), 4)
        avg_values.append(avg)
    df2 = df.assign(Average=avg_values)
    #df2.to_csv(path)
    return df2


def transformer(dataframe, asset_class):

    if asset_class == "bond":
        df = avg_ochl(dataframe)

    elif asset_class == "commodity" or asset_class == "index":
        df = avg_ochl(dataframe)
        df = volume_rm(df)

    elif asset_class == "currency_cross":
        df = avg_ochl(dataframe)
        df = base_currency(df)

    return df




def gdp_parser(data, regions, years):

    df = pd.read_csv(data)

    regions_list = []
    currencies_list = []
    years_list = []
    gdps_list = []

    for idx, row in df.iterrows():
        if row['Country Name'] in regions:
            for year in years:
                regions_list.append(row['Country Name'])
                currencies_list.append("USD")
                years_list.append(year)
                gdps_list.append(row[str(year)])

    gdp_df = pd.DataFrame(
        list(zip(regions_list, currencies_list, years_list, gdps_list)),
        columns=['Region', 'Currency', 'Year', 'GDP'])


    return gdp_df



def total_inflation(data):
    df = pd.read_csv(data)

    total_col = []

    for idx, row in df.iterrows():
        total_sum = 0

        for name, value in df.iteritems():

            total_sum += value

        total_col.append(total_sum)

    df2 = df.assign(Total=total_col)

    return df2




















