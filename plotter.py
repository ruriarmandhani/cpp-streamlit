import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from scipy import stats

#set seaborn style
sns.set_theme()


def pair_plot(df):
    fig = sns.pairplot(df)
    return fig


def dist_plot(df):
    df = df.melt('date', var_name='type', value_name='value')
    fig = sns.displot(data=df, x='value', hue='type', kde=True, 
        height=5, aspect=2, palette='hls')
    return fig


def prob_plot(df):
    fig, ax = plt.subplots()
    ax = stats.probplot(df, fit=True, plot=plt)
    return fig