# -*- coding: utf-8 -*-
"""filter_func.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eaR77-dHInow2fhnUw2Ki8JgjLdI0Bqa
"""

# Install 'Levenshtein' in case you dont have it
# !pip install python-Levenshtein
import Levenshtein
import pandas as pd


# # Import the Dataset
# prices_df_main = pd.read_csv("prices_test.csv")
# prices_df_main.dropna(inplace=True)
# prices_df_main.set_index("Unnamed: 0",inplace=True)
# prices_df_main.index.name= 'date'


def filter_similar(threshold, data):
  fund_names = list(data.columns)
  distance_limit = 7
  remove_dict ={}
  # Find similar fund names
  for fund in fund_names:
      serie_A= pd.Series(list(map(lambda x, y: Levenshtein.distance(x,y),[fund]*len(fund_names) , fund_names)))
      serie_A = serie_A[serie_A>0]
      remove_dict[fund] = list(map(lambda x: fund_names[x], list(serie_A[serie_A<distance_limit].index)))

  # Remove similar fund names
  for key in remove_dict.keys():
    if key in fund_names:
      for name in remove_dict[key]:
        if name in fund_names:
          fund_names.remove(name)
  
  return fund_names


# # Run the function (it takes around 8 mins)
# fund_names = filter_similar(7,prices_df_main)
# # Finally, filter the dataset like this
# prices_df_main[fund_names]


## In case you want to save the filtered list of funds into a pickle
# import pickle
# with open('different_funds_7.pkl', 'wb') as f:
#    pickle.dump(fund_names, f)