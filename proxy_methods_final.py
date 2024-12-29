
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
from modified_spectral_method import *
from modified_louvain_method import *
import cvxpy as cp
import copy
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import ttest_rel
from utils import progress_bar, clear_progress_bar



#-----------------------------------------------------------------------------------------------------------------------------
                                            #COEFFICIENT CALCULATION METHODS
#-----------------------------------------------------------------------------------------------------------------------------
# function to calculate proxy using the intersection method by averages
def calculate_proxy_intersection_method_average(ticker, metadata, communities, ticker_community, prices_data, index_data, liquid_bucket, date, use_index = False):

    if not isinstance(communities[0], list):
        communities = [communities]
        ticker_community = 1

    # prepare the data for the date
    prices_data = prices_data.loc[date,:]

    metadata = metadata.set_index('Ticker')
    metadata_ticker_community = metadata[metadata.index.isin(communities[ticker_community - 1])] #Get the metadata for the tickers in the community
    # remove ticker from the metadata_ticker_community
    metadata_ticker_community = metadata_ticker_community[metadata_ticker_community.index != ticker]

    #Extract the tickers in the community in the same bucket as the ticker of interest
    tickers_bucket_community = metadata_ticker_community[(metadata_ticker_community['Sector'] == metadata.loc[ticker, 'Sector']) & 
                                                         (metadata_ticker_community['Country'] == metadata.loc[ticker, 'Country']) & 
                                                         (metadata_ticker_community['AverageRating'] == metadata.loc[ticker, 'AverageRating'])].index.to_list()

    #Check if there is only one ticker in the bucket
    if len(tickers_bucket_community) == 0:
        #print(f"Bucket not found for Company {ticker}")
        if use_index:
            #Extract spread from index data for the date
            index_data = index_data[index_data['Date'] == date]
            index_spread = index_data['ConvSpread'].values[0] #.values converts column to numpy array. We then get the first element of the array.
            # print(f"Index spread for the date {date} is {index_spread}")
            proxy = index_spread
        else:
            #Remove ticker from metadata so that it is not included in the liquid bucket tickers below
            metadata_1 = metadata[metadata.index != ticker]
            liquid_bucket_tickers = metadata_1[(metadata_1['Sector'] == liquid_bucket['Sector']) & (metadata_1['Country'] == liquid_bucket['Country']) & (metadata_1['AverageRating'] == liquid_bucket['Rating'])].index.to_list()
            liquid_bucket_spread = prices_data[liquid_bucket_tickers].mean(axis=0)
            # print(f"Liquid bucket spread for the date {date} is {liquid_bucket_spread}")
            proxy = liquid_bucket_spread
    else:
        #If there are tickers in the bucket, calculate the average spread for the tickers in the bucket
        proxy = prices_data[tickers_bucket_community].mean(axis=0)

    return proxy

# function to calculate proxy using the intersection method
def calculate_coefficients_intersection_method(prices_data, communities, metadata, index_data, liquid_bucket, date, use_index = False):
    if not isinstance(communities[0], list):
        communities = [communities]
    coefficients = {}
    metadata = metadata.set_index('Ticker')
    # prepare the data for the date
    prices_data = prices_data.loc[date,:]

    # prepare index data for the date
    index_data.rename(columns={'AsOf':'Date'}, inplace=True)

    #Extract spread from index data for the date
    index_data = index_data[index_data['Date'] == date]
    index_spread = index_data['ConvSpread'].values[0]
    # print(f"Index spread for the date {date} is {index_spread}")


    if use_index:
        global_spread = index_spread
    else:
        # prepare the liquid bucket data for the date
        liquid_bucket_sector = liquid_bucket['Sector']
        liquid_bucket_country = liquid_bucket['Country']
        liquid_bucket_ratings = liquid_bucket['Rating']
        liquid_bucket_tickers = metadata[(metadata['Sector'] == liquid_bucket_sector) & (metadata['Country'] == liquid_bucket_country) & (metadata['AverageRating'] == liquid_bucket_ratings)].index.to_list()
        liquid_bucket_spread = prices_data[liquid_bucket_tickers].mean(axis=0)
        # print(f"Liquid bucket spread for the date {date} is {liquid_bucket_spread}")
        global_spread = liquid_bucket_spread

    for community_number, community in enumerate(communities):

        # create all possible combinations of the buckets for the community
        unique_buckets = []

        for i in range(len(community)):
            bucket = f"{metadata.loc[community[i], 'Sector']}, {metadata.loc[community[i], 'Country']}, {metadata.loc[community[i], 'AverageRating']}"
            if bucket not in unique_buckets:
                unique_buckets.append(bucket) #Get the unique buckets for the community

        if not use_index and f'{liquid_bucket_sector}, {liquid_bucket_country}, {liquid_bucket_ratings}' in unique_buckets:
            # remove the liquid bucket from the unique buckets
            unique_buckets.remove(f'{liquid_bucket_sector}, {liquid_bucket_country}, {liquid_bucket_ratings}')

        # prepare the prices data for community and the indicator matrix
        prices_data_community = prices_data[community]
        prices_data_community = prices_data_community.T.to_numpy()
        prices_data_community = prices_data_community.reshape(-1, 1)

        # create the indicator matrix
        indicators = np.zeros((len(community), len(unique_buckets)))
        for i in range(len(community)):
            # create string for the bucket
            bucket = f"{metadata.loc[community[i], 'Sector']}, {metadata.loc[community[i], 'Country']}, {metadata.loc[community[i], 'AverageRating']}"
            if bucket in unique_buckets:
                j = unique_buckets.index(bucket)
                indicators[i, j] = 1
        
        # a_0
        a_0 = np.tile(global_spread, (len(community), 1))

        # create optimization variables
        betas = cp.Variable(shape=(len(unique_buckets), 1))

        beta_contributions = indicators @ betas

        # Define the objective function
        objective = cp.Minimize(cp.norm(prices_data_community - a_0 - beta_contributions, "fro")**2)

        # Solve the optimization problem
        problem = cp.Problem(objective)
        problem.solve()

        # store the coefficients
        coefficients_df = pd.DataFrame({'bucket': unique_buckets, 'Coefficient': betas.value.flatten()})
        coefficients_df = coefficients_df.set_index('bucket')
    
        coefficients[f'community_{community_number+1}'] = coefficients_df
    
    return coefficients


# function to calculate proxy using the CSRA community method
def calculate_proxy_coeff_csra_community(prices_data, communities, metadata, index_data, liquid_bucket, date, use_index = False):
    if not isinstance(communities[0], list):
        communities = [communities]
    coefficients = {}
    metadata = metadata.set_index('Ticker')
    # prepare the data for the date
    prices_data = prices_data.loc[date,:]

    # Extract spread from index data for the date
    index_data = index_data[index_data['Date'] == date]
    index_spread = index_data['ConvSpread'].values[0]
    # print(f"Index spread for the date {date} is {index_spread}")

    # print(f"Liquid bucket spread for the date {date} is {liquid_bucket_spread}")

    if use_index:
        global_spread = index_spread
    else:
        # prepare the liquid bucket data for the date
        liquid_bucket_sector = liquid_bucket['Sector']
        liquid_bucket_country = liquid_bucket['Country']
        liquid_bucket_ratings = liquid_bucket['Rating']
        liquid_bucket_tickers = metadata[(metadata['Sector'] == liquid_bucket_sector) & (metadata['Country'] == liquid_bucket_country) & (metadata['AverageRating'] == liquid_bucket_ratings)].index.to_list()
        liquid_bucket_spread = prices_data[liquid_bucket_tickers].mean(axis=0)
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

        # min_val = prices_data_community.min()
        # max_val = prices_data_community.max()

        # # Min-Max Normalization
        # prices_data_community_normalized = (prices_data_community - min_val) / (max_val - min_val)

        # # Add a small constant to avoid log(0)
        # epsilon = 1e-8  # Small positive constant
        # prices_data_community_normalized += epsilon

        # # Transform the data to the log space
        # prices_data_community_log = np.log(prices_data_community_normalized)

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

        regularization = 1e-4 * cp.norm(betas, 2) #A small value like 1e-41e-4 assumes that the residual error ∥y−Xβ∥22∥y−Xβ∥22​ is numerically dominant and regularization is used primarily to stabilize the solution.
        objective = cp.Minimize(cp.norm(prices_data_community_log - beta_0 - beta_contributions, "fro")**2 + regularization)

        
        # Define the objective function
        #objective = cp.Minimize(cp.norm(prices_data_community_log - beta_0 -  beta_contributions, "fro")**2)

        
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

#-----------------------------------------------------------------------------------------------------------------------------
                                            #PROXY CALCULATION METHODS
#-----------------------------------------------------------------------------------------------------------------------------

#Intersection method
def calculate_proxy_intersection_method(ticker, metadata, coefficients, ticker_community, prices_data, index_data, liquid_bucket, date, use_index = False):
    # if there is only one community, then the ticker_community will be 1
    if len(coefficients) == 1:
        ticker_community = 1

    metadata = metadata.set_index('Ticker')
    # prepare the data for the date
    prices_data = prices_data.loc[date,:]

    #Extract spread from index data for the date
    index_data = index_data[index_data['Date'] == date]
    index_spread = index_data['ConvSpread'].values[0]
    # print(f"Index spread for the date {date} is {index_spread}")

  
    # print(f"Liquid bucket spread for the date {date} is {liquid_bucket_spread}")

    if use_index:
        global_spread = index_spread
    else:
        # prepare the liquid bucket data for the date
        liquid_bucket_sector = liquid_bucket['Sector']
        liquid_bucket_country = liquid_bucket['Country']
        liquid_bucket_ratings = liquid_bucket['Rating']

        #Remove ticker from metadata so that it is not included in the liquid bucket tickers below
        metadata = metadata[metadata['Ticker'] != ticker]

        liquid_bucket_tickers = metadata[(metadata['Sector'] == liquid_bucket_sector) & (metadata['Country'] == liquid_bucket_country) & (metadata['AverageRating'] == liquid_bucket_ratings)].index.to_list()
        liquid_bucket_spread = prices_data[liquid_bucket_tickers].mean(axis=0)
        global_spread = liquid_bucket_spread


    company_bucket = f"{metadata.loc[ticker, 'Sector']}, {metadata.loc[ticker, 'Country']}, {metadata.loc[ticker, 'AverageRating']}"

    company_community = f'community_{ticker_community}'
    coefficients_ticker_community = coefficients[company_community]

    if company_bucket not in coefficients_ticker_community.index:
        #print(f"Bucket not found for Company {ticker} (INTERSECTION METHOD) ")
        return global_spread
    else:
        # print(f"Bucket FOUND for Company {ticker} ")
        coefficient = coefficients_ticker_community.loc[company_bucket, 'Coefficient']

        proxy = coefficient + global_spread

    return proxy

#CSRA method -- Can be used for both community and non-community based methods

def calculate_proxy_csra_community(ticker,  metadata, coefficients, ticker_community, prices_data, index_data, liquid_bucket, date, use_index = False):
    # if there is only one community, then the ticker_community will be 1
    if len(coefficients) == 1:
        ticker_community = 1
    
    metadata = metadata.set_index('Ticker')
    # prepare the data for the date
    prices_data = prices_data.loc[date,:]

    #Extract spread from index data for the date
    index_data = index_data[index_data['Date'] == date]
    index_spread = index_data['ConvSpread'].values[0]
    # print(f"Index spread for the date {date} is {index_spread}")

    # print(f"Liquid bucket spread for the date {date} is {liquid_bucket_spread}")

    if use_index:
        global_spread = np.log(index_spread)
    else:
        # Prepare the liquid bucket data for the date
        liquid_bucket_sector = liquid_bucket['Sector']
        liquid_bucket_country = liquid_bucket['Country']
        liquid_bucket_ratings = liquid_bucket['Rating']

        #Remove ticker from metadata so that it is not included in the liquid bucket tickers below
        metadata_1 = metadata[metadata['Ticker'] != ticker]

        liquid_bucket_tickers = metadata_1[(metadata_1['Sector'] == liquid_bucket_sector) & (metadata_1['Country'] == liquid_bucket_country) & (metadata_1['AverageRating'] == liquid_bucket_ratings)].index.to_list()
        liquid_bucket_spread = prices_data[liquid_bucket_tickers].mean(axis=0)
        global_spread = np.log(liquid_bucket_spread)

    company_community = f'community_{ticker_community}'
    coefficients_ticker_community = coefficients[company_community]

    # get the coefficients for the sector, country and rating
    #Check that the sector, country and rating for the ticker are in the coefficients dataframe. THEY WON'T BE IF THE TICKER COMES FROM SOME UNIDENTIFIED SECTOR, COUNTRY OR RATING 
    if metadata.loc[ticker, 'Sector'] in coefficients_ticker_community.index:
        sector_coefficient = coefficients_ticker_community.loc[metadata.loc[ticker, 'Sector'], 'Coefficient']
    else:
        sector_coefficient = 0
    
    if metadata.loc[ticker, 'Country'] in coefficients_ticker_community.index:
        country_coefficient = coefficients_ticker_community.loc[metadata.loc[ticker, 'Country'], 'Coefficient']
    else:
        country_coefficient = 0

    if metadata.loc[ticker, 'AverageRating'] in coefficients_ticker_community.index:
        rating_coefficient = coefficients_ticker_community.loc[metadata.loc[ticker, 'AverageRating'], 'Coefficient']
    else:
        rating_coefficient = 0

    proxy_log = sector_coefficient + country_coefficient + rating_coefficient + global_spread

    proxy = np.exp(proxy_log)

    return proxy


#-----------------------------------------------------------------------------------------------------------------------------
                                            #GET BUCKET METHOD
#-----------------------------------------------------------------------------------------------------------------------------

def get_bucket(metadata, company_communities, ticker_proxy):
    """
    Get the bucket for a given ticker.

    Parameters:
    - metadata (pd.DataFrame): The metadata dataframe containing ticker information.
    - company_communities (list of lists): A list of communities where each community is a list of tickers.
    - ticker_proxy (str): The ticker for which to calculate the bucket.

    Returns:
    - dict: A dictionary representing the bucket with keys 'Sector', 'Country', and 'Rating', or None if the ticker is not found.
    - int: The community number the ticker belongs to, or None if the ticker is not in any community.
    """
    # Find the community to which the ticker belongs
    ticker_community = None
    for i, community in enumerate(company_communities):
        if ticker_proxy in community:
            ticker_community = i + 1
            break

    if ticker_community is None:
        print(f"Ticker {ticker_proxy} is not part of any community.")
        return None, None

    # Extract the row corresponding to the ticker
    ticker_data = metadata[metadata['Ticker'] == ticker_proxy]

    if ticker_data.empty:
        print(f"Ticker {ticker_proxy} not found in metadata.")
        return None, None

    # Create the liquid bucket using the extracted data
    liquid_bucket = {
        'Sector': ticker_data.iloc[0]['Sector'],
        'Country': ticker_data.iloc[0]['Country'],
        'Rating': ticker_data.iloc[0]['AverageRating']
    }


    return liquid_bucket, ticker_community



#-----------------------------------------------------------------------------------------------------------------------------
                                    #MAIN METHODS (CALCULATIONS AND ANALYSIS)
#-----------------------------------------------------------------------------------------------------------------------------


def calculate_proxies_and_add_to_metadata(metadata, company_communities, prices_data, index_data, liquid_bucket, date):
    """
    Calculates proxy spreads and actual spreads for tickers in metadata and adds them as columns to a copy of metadata.
    
    Parameters:
    - metadata: pd.DataFrame, contains metadata including tickers.
    - company_communities: list of lists, each inner list is a community containing tickers.
    - prices_data: pd.DataFrame, price data with tickers as columns.
    - index_data: additional index-related data.
    - liquid_bucket: dict, liquid bucket information (e.g., Sector, Country, Rating).
    - date: str or pd.Timestamp, the date for which proxies are calculated.

    Returns:
    - metadata_with_proxies: pd.DataFrame, copy of metadata with calculated proxy and actual spread columns added.
    """
    # Ensure date is in pandas datetime format
    date = pd.to_datetime(date, format='%Y-%m-%d')
    
    # Create a copy of metadata to store results
    metadata_with_proxies = metadata.copy()
    metadata_with_proxies['Actual_Spread'] = None
    metadata_with_proxies['Proxy_Intersection'] = None
    metadata_with_proxies['Proxy_Intersection_Community'] = None
    metadata_with_proxies['Proxy_CSRA'] = None
    metadata_with_proxies['Proxy_CSRA_Community'] = None

    # Iterate over tickers in metadata
    for ticker_proxy in metadata['Ticker']:
        #print(f"\nCalculating proxy for ticker: {ticker_proxy}")

        # Find the community to which the ticker belongs
        ticker_community = None
        for i, community in enumerate(company_communities):
            if ticker_proxy in community:
                ticker_community = i + 1
                break

        # Get the actual spread
        actual_spread = prices_data.loc[date, ticker_proxy]
        #print(f"Actual spread for {ticker_proxy}: {actual_spread}")

        # Remove the ticker_proxy from the relevant data structures
        company_communities_proxy_method = copy.deepcopy(company_communities)
        company_communities_proxy_method[ticker_community - 1].remove(ticker_proxy)
        metadata_proxy_method = metadata[metadata['Ticker'] != ticker_proxy]
        prices_data_proxy_method = prices_data.drop(columns=[ticker_proxy])

        # Flatten company communities for proxy method
        all_companies_proxy_method = [company for community in company_communities_proxy_method for company in community]

        # Calculate proxies using intersection and CSRA methods
        try:
            #STANDARD VERSION FOR ALL COMPANIES
            # coefficients_intersection = calculate_coefficients_intersection_method(
            #     prices_data_proxy_method, all_companies_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=True
            # )
            # proxy_intersection = calculate_proxy_intersection_method(
            #     ticker_proxy, metadata, coefficients_intersection, ticker_community, prices_data_proxy_method, index_data, liquid_bucket, date, use_index=True
            # )

            proxy_intersection = calculate_proxy_intersection_method_average(ticker_proxy, metadata, all_companies_proxy_method, ticker_community, prices_data, index_data, liquid_bucket, date, use_index = True)

        except Exception as e:
            print(f"Error calculating proxy using intersection method for {ticker_proxy}: {e}")
            proxy_intersection = None

        try:
            # coefficients_intersection_community = calculate_coefficients_intersection_method(
            #     prices_data_proxy_method, company_communities_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=True
            # )
            # proxy_intersection_community = calculate_proxy_intersection_method(
            #     ticker_proxy, metadata, coefficients_intersection_community, ticker_community, prices_data_proxy_method, index_data, liquid_bucket, date, use_index=True
            # )

            proxy_intersection_community = calculate_proxy_intersection_method_average(ticker_proxy, metadata, company_communities_proxy_method, ticker_community, prices_data, index_data, liquid_bucket, date, use_index = True)


        except Exception as e:
            print(f"Error calculating proxy using intersection community method for {ticker_proxy}: {e}")
            proxy_intersection_community = None

        try:
            #STANDARD VERSION FOR ALL COMPANIES
            coefficients_csra = calculate_proxy_coeff_csra_community(
                prices_data_proxy_method, all_companies_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=True
            )
            proxy_csra = calculate_proxy_csra_community(
                ticker_proxy, metadata, coefficients_csra, ticker_community, prices_data_proxy_method, index_data, liquid_bucket, date, use_index=True
            )
        except Exception as e:
            print(f"Error calculating proxy using CSRA method for {ticker_proxy}: {e}")
            proxy_csra = None

        try:
            #CSRA COMMUNITY VERSION
            coefficients_csra_community = calculate_proxy_coeff_csra_community(
                prices_data_proxy_method, company_communities_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=True
            )
            proxy_csra_community = calculate_proxy_csra_community(
                ticker_proxy, metadata, coefficients_csra_community, ticker_community, prices_data_proxy_method, index_data, liquid_bucket, date, use_index=True
            )
        except Exception as e:
            print(f"Error calculating proxy using CSRA community method for {ticker_proxy}: {e}")
            proxy_csra_community = None
        
        # Update the copied metadata with the results
        metadata_with_proxies.loc[metadata_with_proxies['Ticker'] == ticker_proxy, 'Actual_Spread'] = actual_spread
        metadata_with_proxies.loc[metadata_with_proxies['Ticker'] == ticker_proxy, 'Proxy_Intersection'] = proxy_intersection
        metadata_with_proxies.loc[metadata_with_proxies['Ticker'] == ticker_proxy, 'Proxy_Intersection_Community'] = proxy_intersection_community
        metadata_with_proxies.loc[metadata_with_proxies['Ticker'] == ticker_proxy, 'Proxy_CSRA'] = proxy_csra
        metadata_with_proxies.loc[metadata_with_proxies['Ticker'] == ticker_proxy, 'Proxy_CSRA_Community'] = proxy_csra_community

    return metadata_with_proxies



def calculate_proxy_time_series(
    tickers, metadata, company_communities, prices_data, index_data, dates
):
    """
    Calculates the time series of proxy values for the specified tickers using the different proxy methods. 
    Outputs a final DataFrame with the actual spread, proxy spread using intersection method, proxy spread using CSRA method, and proxy spread using community methods.

    Parameters:
    - ticker: str, the ticker for which to calculate proxies.
    - metadata: pd.DataFrame, metadata including the ticker.
    - company_communities: list of lists, each inner list is a community containing tickers.
    - prices_data: pd.DataFrame, price data with tickers as columns.
    - index_data: additional index-related data.
    - date_range: iterable of dates.

    Returns:
    - pd.DataFrame, time series of proxy values for the specified ticker.
    """
    results = []
 
    for ticker_index,ticker in enumerate(tickers):
        liquid_bucket = get_bucket(metadata, company_communities, ticker)
        for date in dates:
            try:
                # Ensure the date is in pandas datetime format
                date = pd.to_datetime(date)
                
                # Locate the community for the ticker
                ticker_community = None
                for i, community in enumerate(company_communities):
                    if ticker in community:
                        ticker_community = i + 1
                        break
                
                # Get actual spread
                actual_spread = prices_data.loc[date, ticker]
                
                # Remove the ticker from proxy calculations (effectively performing leave-one-out)
                company_communities_proxy_method = copy.deepcopy(company_communities)
                company_communities_proxy_method[ticker_community - 1].remove(ticker) # Remove the ticker from its community list
                metadata_proxy_method = metadata[metadata['Ticker'] != ticker] # Remove the ticker from metadata
                prices_data_proxy_method = prices_data.drop(columns=[ticker]) # Remove the ticker from prices data
                
                # Flatten communities
                all_companies_proxy_method = [
                    company for community in company_communities_proxy_method for company in community
                ]

                # Calculate proxies
                proxy_intersection = None
                proxy_intersection_community = None
                proxy_csra = None
                proxy_csra_community = None

                try:
                    #INTERSECTION METHOD (STANDARD VERSION FOR ALL COMPANIES)
                    #----------------------------------------------------------
                    # coefficients_intersection = calculate_coefficients_intersection_method(
                    #     prices_data_proxy_method, all_companies_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=True
                    # )
                    # proxy_intersection = calculate_proxy_intersection_method(
                    #     ticker, metadata, coefficients_intersection, ticker_community, prices_data_proxy_method, index_data, liquid_bucket, date, use_index=True
                    # )

                    proxy_intersection = calculate_proxy_intersection_method_average(ticker, metadata, all_companies_proxy_method, ticker_community, prices_data, index_data, liquid_bucket, date, use_index = True)

                except Exception as e:
                    print(f"Error calculating proxy using intersection method for {ticker}: {e}")
                    proxy_intersection = None

                try:
                    #INTERSECTION METHOD (COMMUNNITY VERSION)
                    #----------------------------------------------------------
                    # coefficients_intersection_community = calculate_coefficients_intersection_method(
                    #     prices_data_proxy_method, company_communities_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=True
                    # )
                    # proxy_intersection_community = calculate_proxy_intersection_method(
                    #     ticker, metadata, coefficients_intersection_community, ticker_community, prices_data_proxy_method, index_data, liquid_bucket, date, use_index=True
                    # )

                    proxy_intersection_community = calculate_proxy_intersection_method_average(ticker, metadata, company_communities_proxy_method, ticker_community, prices_data, index_data, liquid_bucket, date, use_index = True)

                except Exception as e:
                    print(f"Error calculating proxy using intersection community method for {ticker}: {e}")
                    proxy_intersection_community = None

                
                try:
                    #CSRA METHOD (STANDARD VERSION FOR ALL COMPANIES)
                    #----------------------------------------------------------
                    coefficients_csra = calculate_proxy_coeff_csra_community(
                        prices_data_proxy_method, all_companies_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=True
                    )
                    proxy_csra = calculate_proxy_csra_community(
                        #Here we pass in 1 for the community number since the coefficients will be calculate using all the tickers, and will be stored in coefficients dictionary with key 'community_1'
                        ticker, metadata, coefficients_csra, 1, prices_data_proxy_method, index_data, liquid_bucket, date, use_index=True
                )
                except Exception as e:
                    print(f"Error calculating CSRA NORMAL proxy for {ticker} on {date}: {e}")

                    #CSRA METHOD (COMMUNNITY VERSION)
                    #----------------------------------------------------------
                try:
                    coefficients_csra_community = calculate_proxy_coeff_csra_community(
                        prices_data_proxy_method, company_communities_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=True
                    )
                    proxy_csra_community = calculate_proxy_csra_community(
                        ticker, metadata, coefficients_csra_community, ticker_community, prices_data_proxy_method, index_data, liquid_bucket, date, use_index=True
                    )
                except Exception as e:
                    print(f"Error calculating CSRA COMMUNITY proxy for {ticker} on {date}: {e}")
                
                # Append results for this date
                results.append({
                    "Ticker": ticker,
                    "Date": date,
                    "Actual_Spread": actual_spread,
                    "Proxy_Intersection": proxy_intersection,
                    "Proxy_Intersection_Community": proxy_intersection_community,
                    "Proxy_CSRA": proxy_csra,
                    "Proxy_CSRA_Community": proxy_csra_community,
                })

            except Exception as e:
                print(f"Error calculating proxies for {ticker} on {date}: {e}")
        
        progress_bar(ticker_index / len(tickers))  #progress bar for each ticker
    
    clear_progress_bar()
        # Convert results to a DataFrame
    return pd.DataFrame(results)

def calculate_rmse(actual_spread, proxy_spread):
    """
    Calculate the RMSE between actual spreads and proxy spreads.

    Parameters:
    - actual_spread (pd.Series or np.array): The actual spreads (\( \hat{S}_j \)).
    - proxy_spread (pd.Series or np.array): The proxy spreads (\( S_j^{proxy} \)).

    Returns:
    - float: The RMSE value.
    """
    # Ensure the inputs are numpy arrays
    actual_spread = np.array(actual_spread)
    proxy_spread = np.array(proxy_spread)
    
    # Calculate the squared differences
    squared_differences = (actual_spread - proxy_spread) ** 2
    
    # Calculate the mean of squared differences
    mean_squared_error = np.mean(squared_differences)
    
    # Return the square root of the mean squared error
    return np.sqrt(mean_squared_error)

def calculate_rmse_curves(proxy_time_series_df, dates):

    """
    Calculates the RMSE curves for normal proxy methods and community proxy methods over the given dates

    parameters:
    proxy_time_series_df: pd.DataFrame, time series of actual and proxy values for all tickers.
    dates: list, list of dates over which the cds/price data is available (dates for cds/price data and index data should be the same).
    """

    #Lists for rmse's for normal proxy methods
    rmse_csra_normal_list = []
    rmse_intersection_normal_list = []

    #Lists for rmse's for community proxy methods
    rmse_csra_communities_list = []
    rmse_intersection_communities_list = []

    #Convert the date column to datetime format
    proxy_time_series_df['Date'] = pd.to_datetime(proxy_time_series_df['Date'])


    for date in dates:
        #Filter the proxy time series data by the date        

        Proxy_spreads_filtered = proxy_time_series_df[proxy_time_series_df['Date'] == date]
        
        #Get the actual spreads and the proxy spreads
        actual_spreads = Proxy_spreads_filtered['Actual_Spread']
        proxy_spreads_intersection_normal = Proxy_spreads_filtered['Proxy_Intersection']
        proxy_spreads_intersection_communities = Proxy_spreads_filtered['Proxy_Intersection_Community']
        proxy_spreads_csra_normal = Proxy_spreads_filtered['Proxy_CSRA']
        proxy_spreads_csra_communities = Proxy_spreads_filtered['Proxy_CSRA_Community']



        #Calculate the RMSE for intersection method
        rmse_intersection_normal = calculate_rmse(actual_spreads, proxy_spreads_intersection_normal)
        rmse_intersection_communities = calculate_rmse(actual_spreads, proxy_spreads_intersection_communities)

        rmse_intersection_normal_list.append(rmse_intersection_normal)
        rmse_intersection_communities_list.append(rmse_intersection_communities)
        
        #Calculate the RMSE for CSRA method
        rmse_csra_normal = calculate_rmse(actual_spreads, proxy_spreads_csra_normal)
        rmse_csra_communities = calculate_rmse(actual_spreads, proxy_spreads_csra_communities)

        rmse_csra_normal_list.append(rmse_csra_normal)
        rmse_csra_communities_list.append(rmse_csra_communities)


    return rmse_intersection_normal_list, rmse_intersection_communities_list, rmse_csra_normal_list, rmse_csra_communities_list


def calculate_percentage_better(rmse_csra_normal_list, rmse_csra_communities_list, method = "CSRA"):

    """
    Calculate the percentage of days where CSRA Communities method is better than CSRA Normal method.

    parameters:
    rmse_csra_normal_list: list, A list of RMSE values for CSRA Normal method.
    rmse_csra_communities_list: list, A list of RMSE values for CSRA Communities method.
    """

    rmse_csra_normal_list_np = np.array(rmse_csra_normal_list)
    rmse_csra_communities_list_np = np.array(rmse_csra_communities_list)

    # Find indexes where elements in rmse_csra_normal_list are greater than in rmse_csra_communities_list (inidicates CSRA Communities method is better)
    greater_than_condition = rmse_csra_normal_list_np > rmse_csra_communities_list_np

    # Count the number of True values
    count = np.sum(greater_than_condition)

    #Calculate the percentage
    percentage_better = (count/len(rmse_csra_normal_list)) * 100

    print(f"Number of days where RMSE for normal {method} method > RMSE for community {method} method : {count}")
    print(f"Percentage of days where {method} Communities method is better: {percentage_better}%")  

def paired_t_test(rmse_csra_normal_list, rmse_csra_communities_list):

    #We perform a paired t test since we are testing a statistic for the same sample (the same tickers and proxy methods) over time

    #First check if differences are normally distributed (required for a t test)

    #Calculate the differences
    differences = np.array(rmse_csra_normal_list) - np.array(rmse_csra_communities_list)
    
    level_of_significance = 0.05

    #Perform Shapiro-Wilk test
    stat, p = shapiro(differences)
    print(f"Shapiro-Wilk Test Statistic: {stat}, p-value: {p}")

    print("CHECKING FOR NORMALITY OF DIFFERENCES")
    print("--------------------------------------")

    if p > level_of_significance:
        print("The differences appear to be normally distributed (fail to reject H0).")

        print("Since the differences are normally distributed, we can perform a paired t-test")
        #Perform paired t-test
        t_stat_paired_t_test, p_paired_t_test = ttest_rel(rmse_csra_normal_list, rmse_csra_communities_list)

        print("PAIRED T-TEST RESULTS ")
        print("--------------------------------------")

        print(f"Paired T-Test Statistic: {t_stat_paired_t_test}, p-value: {p_paired_t_test}")

        if (p_paired_t_test/2 < level_of_significance and t_stat_paired_t_test > 0):
            print("Reject the null hypothesis for a one-tailed test: Method 1 has higher RMSE.")
            print("Therefore the average RMSE for the normal CSRA method is significantly greater than the average RMSE for the community CSRA method.")
        else:
            print("Fail to reject the null hypothesis for a one-tailed test: Method 1 has lower RMSE..")

    else: 
        print("The differences do not appear to be normally distributed (reject H0).")
        print("Cannot perform a paired t-test.")

        #NOTES
        #------
        #We divide the p_paired_t_test by two to account for the fact that we are doing a one tail test (by defualt this p-value is calculated for a two tail test)
        #We check if t_stat_paired_t_test to make sure that it corresponds to the positive tail of the distribution, since we are chekcing if the mean method one's RMSE is GREATER than the mean of method two's RMSE
    
#-----------------------------------------------------------------------------------------------------------------------------
                                        #PLOTTING METHODS
#-----------------------------------------------------------------------------------------------------------------------------

# Plotting the time series of the proxy methods to compare them with the actual spread
def plot_proxy_time_series_ticker(proxy_time_series):

    """
    Plot the time series of proxy values for a specific ticker to compare them with the actual spread.

    Parameters:
    Proxy_time_series: pd.DataFrame, time series of proxy values for a specific ticker.
    
    """

    plt.figure(figsize=(12, 6))

    # Plot each proxy method and actual spread
    plt.plot(proxy_time_series['Date'], proxy_time_series['Actual_Spread'], label='Actual Spread')
    plt.plot(proxy_time_series['Date'], proxy_time_series['Proxy_Intersection'], label='Proxy Intersection')
    plt.plot(proxy_time_series['Date'], proxy_time_series['Proxy_Intersection_Community'], label='Proxy Intersection Community')
    plt.plot(proxy_time_series['Date'], proxy_time_series['Proxy_CSRA'], label='Proxy CSRA')
    plt.plot(proxy_time_series['Date'], proxy_time_series['Proxy_CSRA_Community'], label='Proxy CSRA Community')

    # Adding title and labels
    plt.title(f"Proxy Methods for Ticker Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Spread", fontsize=12)

    # Adding grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_rmse_curves(rmse_csra_normal_list, rmse_csra_communities_list, dates, method = "CSRA"):

    """
    Plots the RMSE curves for normal proxy methods and community proxy methods over the given dates.

    parameters:
    rmse_csra_normal_list: list, A list of RMSE values for CSRA Normal method.
    rmse_csra_communities_list: list, A list of RMSE values for CSRA Communities method.
    dates: list, list of dates over which the cds/price data is available (dates for cds/price data and index data should be the same).
    """

    num_days = [i for i in range(len(dates))]

    #Plot RMSE for CSRA normal and CSRA communities
    plt.plot(num_days, rmse_csra_normal_list, label=f'{method} Normal')
    plt.plot(num_days, rmse_csra_communities_list, label=f'{method} Communities')
    plt.xlabel('Days')
    plt.ylabel('RMSE')
    plt.title(f'RMSE for {method} Normal and {method} Communities over {len(dates)} days')
    plt.legend()

      



