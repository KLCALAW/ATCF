import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
from modified_spectral_method import *
from modified_louvain_method import *
from proxy_methods_final import *
import cvxpy as cp
import copy
import matplotlib.pyplot as plt

def calculate_variance(ticker, proxy_time_series):

    # Filter the proxy time series for the ticker
    proxy_time_series_ticker = proxy_time_series[proxy_time_series['Ticker'] == ticker]

    # sort the time series by date

    intersection_errors = proxy_time_series_ticker['Actual_Spread'] - proxy_time_series_ticker['Proxy_Intersection']
    intersection_variance = np.var(intersection_errors)

    intersection_communnity_errors = proxy_time_series_ticker['Actual_Spread'] - proxy_time_series_ticker['Proxy_Intersection_Community']
    intersection_community_variance = np.var(intersection_communnity_errors)

    csra_errors = proxy_time_series_ticker['Actual_Spread'] - proxy_time_series_ticker['Proxy_CSRA']
    csra_variance = np.var(csra_errors)

    csra_community_errors = proxy_time_series_ticker['Actual_Spread'] - proxy_time_series_ticker['Proxy_CSRA_Community']
    csra_community_variance = np.var(csra_community_errors)

    return intersection_variance, intersection_community_variance, csra_variance, csra_community_variance

if __name__ == '__main__':
    proxy_time_series_final = pd.read_csv('data/proxy_time_series_final.csv')

    # Calculate the variance of the proxy time series for a ticker
    ticker = 'AAUK'

    intersection_variance, intersection_community_variance, csra_variance, csra_community_variance = calculate_variance(ticker, proxy_time_series_final)

    print(f'Intersection Variance: {intersection_variance}')
    print(f'Intersection Community Variance: {intersection_community_variance}')
    print(f'CSRA Variance: {csra_variance}')
    print(f'CSRA Community Variance: {csra_community_variance}')

    metadata = pd.read_csv('data/metadata.csv')

    variance_df = pd.DataFrame(columns=['Ticker', 'Intersection Variance', 'Intersection Community Variance', 'CSRA Variance', 'CSRA Community Variance'])

    for ticker in metadata['Ticker']:
        
        intersection_variance, intersection_community_variance, csra_variance, csra_community_variance = calculate_variance(ticker, proxy_time_series_final)
        
        new_row = pd.DataFrame([{'Ticker': ticker, 'Intersection Variance': intersection_variance, 'Intersection Community Variance': intersection_community_variance, 'CSRA Variance': csra_variance, 'CSRA Community Variance': csra_community_variance}])

        variance_df = pd.concat([variance_df, new_row], ignore_index=True)

    variance_df.to_csv('data/proxy_variance.csv', index=False)