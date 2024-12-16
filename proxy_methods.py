import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
from modified_spectral_method import *
from modified_louvain_method import *
import cvxpy as cp

# function to calculate proxy using the intersection method
def calculate_coefficients_intersection_method(prices_data, company_communities, metadata, index_data, liquid_bucket, date, use_index = False):

    metadata = metadata.set_index('Ticker')
    # company_communities = metadata.index.to_list()
    # prepare the data for the date
    prices_data = prices_data.loc[date,:]

    # prepare index data for the date
    index_data.rename(columns={'AsOf':'Date'}, inplace=True)
    index_data['Date'] = pd.to_datetime(index_data['Date'], format='%d/%b/%y')
    index_data = index_data.sort_values(by='Date', ascending=True)
    index_data = index_data[index_data['Date'] == date]
    index_spread = index_data['ConvSpread'].values[0]
    print(f"Index spread for the date {date} is {index_spread}")

    # prepare the liquid bucket data for the date
    liquid_bucket_sector = liquid_bucket['Sector']
    liquid_bucket_country = liquid_bucket['Country']
    liquid_bucket_ratings = liquid_bucket['Rating']
    liquid_bucket_tickers = metadata[(metadata['Sector'] == liquid_bucket_sector) & (metadata['Country'] == liquid_bucket_country) & (metadata['AverageRating'] == liquid_bucket_ratings)].index.to_list()
    liquid_bucket_spread = prices_data[liquid_bucket_tickers].mean(axis=0)
    print(f"Liquid bucket spread for the date {date} is {liquid_bucket_spread}")

    if use_index:
        global_spread = index_spread
    else:
        global_spread = liquid_bucket_spread

    # create unique buckets
    unique_sectors = metadata['Sector'].unique().tolist()
    unique_countries = metadata['Country'].unique().tolist()
    unique_ratings = metadata['AverageRating'].unique().tolist()

    if not use_index:
        unique_sectors.remove(liquid_bucket_sector)
        unique_countries.remove(liquid_bucket_country)
        unique_ratings.remove(liquid_bucket_ratings)

    # create all possible combinations of the buckets
    unique_buckets = []

    for i in range(len(company_communities)):
        bucket = f'{metadata.loc[company_communities[i], 'Sector']}, {metadata.loc[company_communities[i], 'Country']}, {metadata.loc[company_communities[i], 'AverageRating']}'
        if bucket not in unique_buckets:
            unique_buckets.append(bucket)
    if not use_index:
        unique_buckets.remove(f'{liquid_bucket_sector}, {liquid_bucket_country}, {liquid_bucket_ratings}')

    prices_data = prices_data.T.to_numpy()
    prices_data = prices_data.reshape(-1, 1)

    # create the masks
    mask = np.zeros((len(company_communities), len(unique_buckets)))
    for i in range(len(company_communities)):
        # create string for the bucket
        bucket = f'{metadata.loc[company_communities[i], 'Sector']}, {metadata.loc[company_communities[i], 'Country']}, {metadata.loc[company_communities[i], 'AverageRating']}'
        if bucket in unique_buckets:
            j = unique_buckets.index(bucket)
            mask[i, j] = 1
    mask_df = pd.DataFrame(mask)
    mask_df.to_csv('mask.csv')

    # beta_0
    beta_0 = np.tile(global_spread, (len(company_communities), 1))

    # create optimization variables
    betas = cp.Variable(shape=(len(unique_buckets), 1))

    beta_contributions = mask @ betas

    # Define the objective function
    objective = cp.Minimize(cp.norm(prices_data - beta_0 - beta_contributions, "fro")**2)

    # Solve the optimization problem
    problem = cp.Problem(objective)
    problem.solve()

    # store the coefficients
    coefficients = pd.DataFrame({'bucket': unique_buckets, 'Coefficient': betas.value.flatten()})

    return coefficients

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


def calculate_proxy_coeff_csra_community(prices_data, communities, metadata, index_data, liquid_bucket, date, use_index = False):
    if not isinstance(communities[0], list):
        communities = [communities]
    coefficients = {}
    metadata = metadata.set_index('Ticker')
    # prepare the data for the date
    prices_data = prices_data.loc[date,:]

    # prepare index data for the date
    index_data.rename(columns={'AsOf':'Date'}, inplace=True)
    index_data['Date'] = pd.to_datetime(index_data['Date'], format='%d/%b/%y')
    index_data = index_data.sort_values(by='Date', ascending=True)
    index_data = index_data[index_data['Date'] == date]
    index_spread = index_data['ConvSpread'].values[0]
    print(f"Index spread for the date {date} is {index_spread}")

    # prepare the liquid bucket data for the date
    liquid_bucket_sector = liquid_bucket['Sector']
    liquid_bucket_country = liquid_bucket['Country']
    liquid_bucket_ratings = liquid_bucket['Rating']
    liquid_bucket_tickers = metadata[(metadata['Sector'] == liquid_bucket_sector) & (metadata['Country'] == liquid_bucket_country) & (metadata['AverageRating'] == liquid_bucket_ratings)].index.to_list()
    liquid_bucket_spread = prices_data[liquid_bucket_tickers].mean(axis=0)

    print(f"Liquid bucket spread for the date {date} is {liquid_bucket_spread}")

    if use_index:
        global_spread = index_spread
    else:
        global_spread = liquid_bucket_spread
    
    for community_number, community in enumerate(communities):

        # prepare the data for the community
        prices_data_community = prices_data[community]
        metadata_community = metadata.loc[community,:]

        sectors_community = metadata_community.loc[community, 'Sector'].unique().tolist()

        countries_community = metadata_community.loc[community, 'Country'].unique().tolist()

        ratings_community = metadata_community.loc[community, 'AverageRating'].unique().tolist()

        # remove the liquid bucket from the community
        if not use_index:
            if liquid_bucket_sector in sectors_community:
                sectors_community.remove(liquid_bucket_sector)
            if liquid_bucket_country in countries_community:
                countries_community.remove(liquid_bucket_country)
            if liquid_bucket_ratings in ratings_community:
                ratings_community.remove(liquid_bucket_ratings)

        prices_data_community = prices_data_community.T.to_numpy()
        prices_data_community = prices_data_community.reshape(-1, 1)

        # tranform the data to the log space
        prices_data_community_log = np.log(prices_data_community)
        # create the masks
        mask = np.zeros((prices_data_community.shape[0], len(sectors_community) + len(countries_community) + len(ratings_community)))

        for i in range(prices_data_community.shape[0]):
            if not use_index:
                if metadata_community.loc[community[i], 'Sector'] in sectors_community:
                    j = sectors_community.index(metadata_community.loc[community[i], 'Sector'])
                    mask[i, j] = 1
                if metadata_community.loc[community[i], 'Country'] in countries_community:
                    j = len(sectors_community) + countries_community.index(metadata_community.loc[community[i], 'Country']) - 1
                    mask[i, j] = 1
                if metadata_community.loc[community[i], 'AverageRating'] in ratings_community:
                    j = len(sectors_community) + len(countries_community) + ratings_community.index(metadata_community.loc[community[i], 'AverageRating']) - 1
                    mask[i, j] = 1


            else:    
                j = sectors_community.index(metadata_community.loc[community[i], 'Sector'])
                mask[i, j] = 1
                k = len(sectors_community) + countries_community.index(metadata_community.loc[community[i], 'Country'])  - 1
                mask[i, k] = 1
                l = len(sectors_community) + len(countries_community) + ratings_community.index(metadata_community.loc[community[i], 'AverageRating']) - 1

        # beta_0
        beta_0 = np.tile(np.log(global_spread), (prices_data_community.shape[1], 1))

        # Define optimization variables
        betas = cp.Variable(shape=(len(sectors_community) + len(countries_community) + len(ratings_community), 1))

        beta_contributions = mask @ betas

        # Define the objective function
        objective = cp.Minimize(cp.norm(prices_data_community_log - beta_0 -  beta_contributions, "fro")**2)


        # Solve the optimization problem
        problem = cp.Problem(objective)
        problem.solve()

        # store the coefficients
        sector_betas = betas.value[0:len(sectors_community)]
        country_betas = betas.value[len(sectors_community): len(sectors_community) + len(countries_community)]
        rating_betas = betas.value[len(sectors_community) + len(countries_community):]

        sectors_df = pd.DataFrame({'Name': sectors_community, 'Coefficient': sector_betas.flatten(), 'Type': 'Sector'},)
        countries_df = pd.DataFrame({'Name': countries_community, 'Coefficient': country_betas.flatten(), 'Type': 'Country'})
        ratings_df = pd.DataFrame({'Name': ratings_community, 'Coefficient': rating_betas.flatten(), 'Type': 'Rating'}) 

        # Combine DataFrames
        coefficients_df = pd.concat([sectors_df, countries_df, ratings_df], ignore_index=True)
        coefficients_df = coefficients_df.set_index('Name')
        coefficients[f'community_{community_number+1}'] = coefficients_df
    
    return coefficients

if __name__ == "__main__":
    prices_data = pd.read_csv('data/reshaped_data.csv')
    prices_data['Date'] = pd.to_datetime(prices_data['Date'], format='%Y-%m-%d')
    prices_data = prices_data.set_index('Date')
    metadata = pd.read_csv('data/metadata.csv')
    # metadata = metadata.set_index('Ticker')  
    correlation_matrix,T,N,company_names = create_correlation_matrix('data/eur_data_standardized_returns.csv')
    C_g = calculate_C_g(correlation_matrix, T, N)
    result_communities, company_communities, modularities = recursive_spectral_method(C_g, correlation_matrix, company_names, min_size=2, modularity_threshold=0.00001)
    print(f"number of communities detected:{len(company_communities)}")
    ticker_proxy = company_communities[len(company_communities)-1][6]
    print(f"Calculating proxy for ticker {ticker_proxy}")
    print(f'The ticker {ticker_proxy} belongs to the community {len(company_communities)}')

    index_data = pd.read_csv('ITRAXX-Europe Timeseries 20241127.csv')
    liquid_bucket = {'Sector': 'Financials', 'Country': 'United Kingdom', 'Rating': 6}
    date = '2015-12-30'
    # convert the date to pd datetime
    date = pd.to_datetime(date, format='%Y-%m-%d')
    
    all_companies = [company for community in company_communities for company in community]

    # calculate the coefficients using the intersection method
    coefficients = calculate_coefficients_intersection_method(prices_data, metadata['Ticker'].to_list(), metadata, index_data, liquid_bucket, date, use_index = False)
    print(coefficients)

    # # calculate the coefficients using the CSRA method
    # coefficients = calculate_proxy_coeff_csra_community(prices_data, company_communities, metadata, index_data, liquid_bucket, date, use_index = True)
    # for key, value in coefficients.items():
    #     print(key)
    #     print(value)
    #     print("")


