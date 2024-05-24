import pandas as pd
import numpy as np


def dataLoader(path="../data/web-redditEmbeddings-subreddits.csv"):
    """
    Load the data from the csv file
    :param path: path to the csv file
    :return: pandas dataframe
    """
    return pd.read_csv(path, header=None)

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