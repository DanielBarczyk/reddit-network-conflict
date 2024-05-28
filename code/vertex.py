import pandas as pd
import numpy as np


def dataLoader(path="../data/web-redditEmbeddings-subreddits.csv"):
    """
    Load the data from the csv file
    :param path: path to the csv file
    :return: pandas dataframe
    """
    return pd.read_csv(path, header=None)

def verticesFromGraph(path="../data/soc-redditHyperlinks-body.tsv"):
    """
    Extract the vertices from the graph
    :param path: path to the tsv file
    :return: pandas dataframe
    """
    graph = pd.read_csv(path, sep='\t')
    
    subreddits = pd.concat([graph['SOURCE_SUBREDDIT'], graph['TARGET_SUBREDDIT']])
    # pair the subreddits with number of occurences (return the dataframe with the counts)
    subreddits = subreddits.value_counts().reset_index()
    # rename the columns
    subreddits.columns = ['subreddit', 'count']
    
    return subreddits

def edgesFromGraph(path="../data/soc-redditHyperlinks-body.tsv"):
    """
    Extract the edges from the graph
    :param path: path to the tsv file
    :return: pandas dataframe
    """
    graph = pd.read_csv(path, sep='\t')
    return graph
# dimension reducing functions

def TSNE(data, n_components=2):
    """
    Reduce the dimension of the data using TSNE
    :param data: pandas dataframe
    :param n_components: number of components to reduce to
    :return: numpy array
    """
    from sklearn.manifold import TSNE
    return TSNE(n_components=n_components).fit_transform(data)

def PCA(data, n_components=2):
    """
    Reduce the dimension of the data using PCA
    :param data: pandas dataframe
    :param n_components: number of components to reduce to
    :return: numpy array
    """
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components).fit_transform(data)

def MDS(data, n_components=2):
    """
    Reduce the dimension of the data using MDS
    :param data: pandas dataframe
    :param n_components: number of components to reduce to
    :return: numpy array
    """
    from sklearn.manifold import MDS
    return MDS(n_components=n_components).fit_transform(data)

def Isomap(data, n_components=2):
    """
    Reduce the dimension of the data using Isomap
    :param data: pandas dataframe
    :param n_components: number of components to reduce to
    :return: numpy array
    """
    from sklearn.manifold import Isomap
    return Isomap(n_components=n_components).fit_transform(data)

if(__name__ == "__main__"):
    verticesFromGraph()