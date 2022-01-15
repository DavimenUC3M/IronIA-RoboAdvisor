import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.decomposition._pca import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import seaborn as sn
import os
import re


def calculate_cluster_variables(start,end,colnames,terminations,dfs,gdps,inflation,verbose=False,column_name="Average"): 

    """
    start: first date to take into account
    end: last date to take into account
    colnames: name columns of the dataframe
    terminations: str added to the end of the names of the df columns 
    dfs: colecction of dataframes to create the variables
    verbose: if True, some prints will appear
    column_name: name of the column use on the mayority of the dataframes

    """

    # offsets = []
    # for i in terminations:
    #     s = ([float(s) for s in re.findall(r'-?\d+\.?\d*', i)]) #Regular expression to get just the numbers
    #     offsets.append(int(s[0]))


    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    clustering_df = pd.DataFrame(columns=colnames)
    
    for rows in range(num_months+1):
        start_date_offset = start_date + relativedelta(months=rows)
        year = str(start_date_offset.year)
        month = str(start_date_offset.month)
        if verbose:
            print(start_date_offset)
        row = []
        for df in dfs:
            offsets = [12,9,6,3,2,1] #This offsets must follow the terminations
            weights = np.array([0.07,0.09,0.12,0.15,0.22,0.35]) #Weights of ponderated mean
            variances = []
            changes = []
            difference = []
            for offset in offsets:

                offset_date = start_date_offset - relativedelta(months=offset)
                variances.append(df[offset_date.isoformat():start_date_offset.isoformat()][column_name].var())
                #changes.append(df[offset_date.isoformat():start_date_offset.isoformat()][column_name].pct_change().mean())
                difference.append(df[offset_date.isoformat():start_date_offset.isoformat()][column_name].diff().mean())

            if len(variances) == len(weights):
                variances = np.array(variances)
                row.append(sum(variances*weights))
                #row.append(sum(changes*weights))
                row.append(sum(difference*weights))

        for gdp in gdps:
            row.append(gdp[year]["GDP"].values[0])

        row.append(inflation[f"{year}-{month}"]["Inflation"].values[0])

        row.append(str(start_date_offset.isoformat()))
        
        clustering_df.loc[len(clustering_df)] = row

    clustering_df["Date"] = pd.to_datetime(clustering_df["Date"])
    clustering_df.set_index("Date",inplace=True)
    
    return clustering_df

def plot_elbow_graph(df,use_PCA=False,n_pca_components=2,verbose=False,n_clusters=10):

    if use_PCA:
        pca = PCA(n_components=n_pca_components)
        pca.fit(df)
        df = pca.transform(df)
    
    num_clusters = range(1, n_clusters+1)
    kmeans = [KMeans(n_clusters=i) for i in num_clusters]
    score = np.array([kmeans[i].fit(df).score(df) for i in range(len(kmeans))])*-1
    if verbose:
        plt.plot(num_clusters,score)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title('Elbow Curve')
        plt.show()

def make_clustering(df,n_clusters,use_PCA=False,n_pca_components=2,verbose=False):

    pca = None

    if use_PCA:
        pca = PCA(n_components=n_pca_components)
        pca.fit(df)
        df = pca.transform(df)

    kmeans = KMeans(n_clusters=n_clusters).fit(df)
    centroids = kmeans.cluster_centers_

    labels = kmeans.predict(df)

    if verbose:
        plt.scatter(centroids[:,0] , centroids[:,1] , s = 300, marker="*")
        
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('Clustering 2D Graph')
        scatter = plt.scatter(df[:,0],df[:,1], c = labels)
        labels_ = np.unique(labels)
        handles = [plt.Line2D([],[],marker="o", ls="", color=scatter.cmap(scatter.norm(yi))) for yi in labels_]
        legends_ = plt.legend(["Centroids"],loc=(0.921,0.76))
        plt.legend(handles, labels_, title="Cluster Label",loc="upper right")
        plt.gca().add_artist(legends_)

    return([kmeans,pca,labels]) #Returns the fitted kmeans, the pca projection and the labels of the train data