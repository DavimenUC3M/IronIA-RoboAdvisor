import investpy
import sys
import numpy as np
import pandas as pd


def investingcom_kwargsgen(asset_type, asset_name, start, end, country, as_json, order, interval):
    allowed_asset_types = ["index", "stock", "commodity", "bond", "currency_cross"]

    if asset_type not in allowed_asset_types:
        sys.exit(f"{asset_type} is not a valid asset type. Aborting scraper.")

    if asset_type == "index" or asset_type == "stock" or asset_type == "commodity":

        args = {
            str(asset_type): str(asset_name),
            "country": country,
            "from_date": start,
            "to_date": end,
            "as_json": as_json,
            "order": order,
            "interval": interval
        }

    elif asset_type == "bond" or asset_type == "currency_cross":

        args = {
            str(asset_type): str(asset_name),
            "from_date": start,
            "to_date": end,
            "as_json": as_json,
            "order": order,
            "interval": interval
        }

    return args


def investingcom_historical_data(asset_type, asset_name,
                                 start="01/01/2000", end="31/12/2020", country=None, as_json=False, order='ascending',
                                 interval='Daily'):
    kwargs = investingcom_kwargsgen(asset_type, asset_name, start, end, country, as_json, order, interval)

    if asset_type == "commodity":
        df = investpy.commodities.get_commodity_historical_data(**kwargs)

    elif asset_type == "bond":
        df = investpy.bonds.get_bond_historical_data(**kwargs)

    elif asset_type == "currency_cross":
        df = investpy.currency_crosses.get_currency_cross_historical_data(**kwargs)

    elif asset_type == "index":
        df = investpy.indices.get_index_historical_data(**kwargs)

    return df
