import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
from modified_spectral_method import *
from modified_louvain_method import *
import cvxpy as cp

# function to calculate proxy using the intersection method
def calculate_proxy_intersection_method(ticker, prices_data, communities, metadata):
    # identify the bucket to which the ticker belongs
    sector = metadata.loc[ticker]['Sector']
    region = metadata.loc[ticker]['Region']
    country = metadata.loc[ticker]['Country']
    # select a liquid bucket which is the average of government sector CDS
    liquid_bucket_tickers = metadata[(metadata['Sector'] == 'Government')].index.to_list()
    # calculate the average of the liquid bucket CDS
    liquid_bucket_prices = prices_data[liquid_bucket_tickers]
    a_0 = liquid_bucket_prices.mean(axis=1)
    # get the bucket to which the ticker belongs
    ticker_bucket = metadata[(metadata['Sector'] == sector) & (metadata['Region'] == region) & (metadata['Country'] == country)].index.to_list()
    # calculate the average of the ticker bucket CDS
    ticker_bucket_prices = prices_data[ticker_bucket]
    a_j = ticker_bucket_prices.mean(axis=1)
    # calculate the intersection proxy
    proxy = a_0 + a_j
    return proxy

def calculate_coefficients_csra_community_based(prices_data, communities, metadata, negative_coefficients=True):
    if not isinstance(communities[0], list):
        communities = [communities]
    coefficients = {}
    unique_sectors = metadata['Sector'].unique().tolist().remove('Government')
    unique_countries = metadata['Country'].unique()
    for community_number, community in enumerate(communities):
        # prepare the data for the community
        prices_data_community = prices_data[community]
        metadata_community = metadata.loc[community,:]

        sectors_community = metadata_community.loc[community, 'Sector'].unique().tolist()

        # use all the governments data as the intercept for the community instead of iTraxx index
        governments_data_community = prices_data[metadata[metadata['Sector'] == 'Government'].index.to_list()].mean(axis=1).to_numpy()
        governments_data_community = np.tile(governments_data_community, (prices_data_community.shape[1], 1))

        countries_community = metadata_community.loc[community, 'Country'].unique().tolist()

        # Prepare the sectors data
        sectors_data_community = pd.DataFrame()
        for sector in sectors_community:
            sectors_data_community[sector] = prices_data_community[metadata_community[metadata_community['Sector'] == sector].index.to_list()].mean(axis=1)
        sectors_data_community = sectors_data_community.T.to_numpy()

        # Prepare the countries data
        countries_data_community = pd.DataFrame()
        for country in countries_community:
            countries_data_community[country] = prices_data_community[metadata_community[(metadata_community['Country'] == country)].index.to_list()].mean(axis=1)
        countries_data_community = countries_data_community.T.to_numpy()

        prices_data_community = prices_data_community.T.to_numpy()
        # tranform the data to the log space
        prices_data_community_log = np.log(prices_data_community)
        sectors_data_community_log = np.log(sectors_data_community)
        countries_data_community_log = np.log(countries_data_community)
        governments_data_community_log = np.log(governments_data_community)

        # create the masks
        mask_sectors = np.zeros((prices_data_community.shape[0], sectors_data_community.shape[0]))
        mask_countries = np.zeros((prices_data_community.shape[0], countries_data_community.shape[0]))

        for i in range(prices_data_community.shape[0]):
            j = sectors_community.index(metadata_community.loc[community[i], 'Sector'])
            mask_sectors[i, j] = 1
        
        for i in range(prices_data_community.shape[0]):
            j = countries_community.index(metadata_community.loc[community[i], 'Country'])
            mask_countries[i, j] = 1

        # Define optimization variables
        row_vector_sectors = cp.Variable(sectors_data_community.shape[0])
        row_vector_countries = cp.Variable(countries_data_community.shape[0])

        # Create a tiled matrix with the same values in all rows
        tiled_matrix_sectors = cp.vstack([row_vector_sectors] * prices_data_community.shape[0])
        tiled_matrix_countries = cp.vstack([row_vector_countries] * prices_data_community.shape[0])

        # Apply the mask using the Hadamard product
        masked_matrix_sectors = cp.multiply(tiled_matrix_sectors, mask_sectors)
        masked_matrix_countries = cp.multiply(tiled_matrix_countries, mask_countries)

        # Compute the contribution from the sectors and countries
        contribution_sectors = masked_matrix_sectors @ sectors_data_community_log
        contribution_countries = masked_matrix_countries @ countries_data_community_log

        # Define the objective function
        objective = cp.Minimize(cp.norm(prices_data_community_log - contribution_sectors - contribution_countries - governments_data_community_log, "fro")**2)

        # Constraints to ensure positive values
        constraints = [row_vector_sectors >= 0, row_vector_countries >= 0]

        # Solve the optimization problem
        if negative_coefficients:
            problem = cp.Problem(objective)
        else:
            problem = cp.Problem(objective, constraints)
        problem.solve()

        # store the coefficients
        sectors_df = pd.DataFrame({'Name': sectors_community, 'Coefficient': row_vector_sectors.value, 'Type': 'Sector'},)
        countries_df = pd.DataFrame({'Name': countries_community, 'Coefficient': row_vector_countries.value, 'Type': 'Country'}) 
        # Combine both DataFrames
        coefficients_df = pd.concat([sectors_df, countries_df], ignore_index=True)
        coefficients_df = coefficients_df.set_index('Name')
        coefficients[f'community_{community_number+1}'] = coefficients_df

    return coefficients

if __name__ == "__main__":
    prices_data = pd.read_csv('reshaped_data.csv')
    prices_data = prices_data.set_index('Date')
    metadata = pd.read_csv('metadata.csv')
    metadata = metadata.set_index('Ticker')
    correlation_matrix, T, N, tickers = create_correlation_matrix('eur_data_standardized_returns.csv')  
    C_g = calculate_C_g(correlation_matrix, T, N)
    result_communities, company_communities, modularities = recursive_spectral_method(C_g, tickers, min_size=2, modularity_threshold=0.00001)
    print(f"number of communities detected:{len(company_communities)}")
    ticker_proxy = company_communities[len(company_communities)-1][6]
    print(f"Calculating proxy for ticker {ticker_proxy}")
    print(f'The ticker {ticker_proxy} belongs to the community {len(company_communities)} and sector {metadata.loc[ticker_proxy]["Sector"]} and country {metadata.loc[ticker_proxy]["Region"]}')

    # calculate the proxy using the intersection method
    proxy = calculate_proxy_intersection_method(ticker_proxy, prices_data, company_communities, metadata)
    correlation = np.corrcoef(prices_data[ticker_proxy], proxy)[0, 1]
    print(f"Correlation between the proxy and the ticker {ticker_proxy} is {correlation}")
    # calculate the RMSE between the proxy and the ticker
    rmse = np.sqrt(np.mean((prices_data[ticker_proxy] - proxy)**2))
    print(f"RMSE between the proxy and the ticker {ticker_proxy} is {rmse}")

    # calculate the coefficients using the CSRA method
    coefficients = calculate_coefficients_csra_community_based(prices_data, company_communities, metadata, negative_coefficients=True)
    for key, value in coefficients.items():
        print(key)
        print(value)
        print("")


