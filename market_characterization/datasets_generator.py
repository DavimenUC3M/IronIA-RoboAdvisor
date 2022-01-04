import investpy
import numpy as np
import pandas as pd
from scrapers import *
from transformations import *


def get_investing_csv(financials, asset_type):

    for asset, country in financials.items():
        asset_class = asset_type
        data = investingcom_historical_data(asset_type=asset_type, asset_name=asset, country=country)
        data = transformer(data, asset_class)

        if country is not None:
            country = "".join(country.split())
            name = "".join(asset.split(" "))
            file = f"{asset_class}_{country}_{name}_data.csv"

        else:
            if asset_class == "currency_cross":
                name = "_".join(asset.split("/"))
                asset_class = "forex"

            elif asset_class == "bond":
                name = "_".join(asset.split(" "))
                name = "".join(name.split("."))

            else:
                name = "".join(asset.split(" "))

            file = f"{asset_class}_{name}_data.csv"

        data.to_csv(f"market_characterization/final_data/{file}")




# ------- Indices final_data -------
indices = {"MSCI World": "world",
           "MSCI World Energy": "world",
           "MSCI World Financials": "world",
           "MSCI World Health Care": "world",
           "MSCI World Industrials": "world",
           "MSCI World Telecom": "world",
           "MSCI World Utilities": "world",
           "MSCI World Consumer Disc": "world",
           "Bloomberg Agriculture": "world",
           "Bloomberg Brent Crude": "world",
           "Bloomberg Natural Gas": "world",
           "MSCI World Large Cap": "world",
           "MSCI World Mid Cap": "world",
           "MSCI World Small Cap": "world",
           "MSCI AC Americas": "world",
           "MSCI AC Asia": "world",
           "MSCI AC Pacific": "world",
           "MSCI AC Europe & Middle East": "world",
           "MSCI AC Far East": "world",
           "DJ US": "united states",
           "Nasdaq": "united states",
           "S&P 500": "united states",
           "CBOE Vix Volatility": "united states",
           "10-year US Treasury VIX": "united states",
           "EuroMTS Eurozone IG 7-10Y Governmen": "euro zone",
           }
# get_investing_csv(indices, "index")



# ------- Commodities final_data -------
commodities = {"Gold": None,
               "Silver": None}
# get_investing_csv(commodities, "commodity")



# ------- Forex final_data -------
forex = {
    "USD/AUD": None,
    "USD/CAD": None,
    "USD/EUR": None,
    "USD/CHF": None,
    "USD/CNY": None,
    "USD/INR": None,
    "USD/GBP": None,
    "USD/JPY": None
}
# get_investing_csv(forex, "currency_cross")



# ------- Bonds final_data -------
bonds = {
    "Canada 5Y": None,
    "France 5Y": None,
    "Germany 5Y": None,
    "Italy 5Y": None,
    "Japan 5Y": None,
    "U.K. 5Y": None,
    "U.S. 5Y": None,
    "India 5Y": None,
    "China 5Y": None
}
# get_investing_csv(bonds, "bond")



# ------- GDP final_data -------

gdp_raw = "market_characterization/raw_data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3469429.csv"
gdp_regions = [
    "Africa Eastern and Southern",
    "Latin America & the Caribbean",
    "United States",
    "Africa Western and Central",
    "East Asia & Pacific",
    "Europe & Central Asia",
]
gdp_years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

# gdp_data = gdp_parser(gdp_raw, gdp_regions, gdp_years).to_csv("market_characterization/final_data/gdp_world_6regions_data.csv")




# ------- Inflation final_data -------

inflation_raw = "market_characterization/raw_data/usa_inflation_USLaborDepartment.csv"

# inflation_data = total_inflation(inflation_raw).to_csv("market_characterization/final_data/inflation_USA.csv")

