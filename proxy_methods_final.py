
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


#-----------------------------------------------------------------------------------------------------------------------------
                                            #COEFFICIENT CALCULATION METHODS
#-----------------------------------------------------------------------------------------------------------------------------

# function to calculate proxy using the intersection method
def calculate_coefficients_intersection_method(prices_data, metadata, index_data, liquid_bucket, date, use_index = False):

    company_communities = metadata['Ticker'].to_list()
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
        print(f"Liquid bucket spread for the date {date} is {liquid_bucket_spread}")
        global_spread = liquid_bucket_spread

    # create all possible combinations of the buckets
    unique_buckets = []

    
    for i in range(len(company_communities)):
        bucket = f"{metadata.loc[company_communities[i], 'Sector']}, {metadata.loc[company_communities[i], 'Country']}, {metadata.loc[company_communities[i], 'AverageRating']}"
        if bucket not in unique_buckets:
            unique_buckets.append(bucket)
    if not use_index:
        # remove the liquid bucket from the unique buckets
        unique_buckets.remove(f'{liquid_bucket_sector}, {liquid_bucket_country}, {liquid_bucket_ratings}')

    # prepare the prices data and the indicator matrix
    prices_data = prices_data.T.to_numpy()
    prices_data = prices_data.reshape(-1, 1)

    # create the indicator matrix
    indicators = np.zeros((len(company_communities), len(unique_buckets)))
    for i in range(len(company_communities)):
        # create string for the bucket
        bucket = f"{metadata.loc[company_communities[i], 'Sector']}, {metadata.loc[company_communities[i], 'Country']}, {metadata.loc[company_communities[i], 'AverageRating']}"
        if bucket in unique_buckets:
            j = unique_buckets.index(bucket)
            indicators[i, j] = 1

    # a_0
    a_0 = np.tile(global_spread, (len(company_communities), 1))

    # create optimization variables
    betas = cp.Variable(shape=(len(unique_buckets), 1))

    beta_contributions = indicators @ betas

    # Define the objective function
    objective = cp.Minimize(cp.norm(prices_data - a_0 - beta_contributions, "fro")**2)

    # Solve the optimization problem
    problem = cp.Problem(objective)
    problem.solve()

    # store the coefficients
    coefficients = pd.DataFrame({'bucket': unique_buckets, 'Coefficient': betas.value.flatten()})
    coefficients = coefficients.set_index('bucket')

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

#-----------------------------------------------------------------------------------------------------------------------------
                                            #PROXY CALCULATION METHODS
#-----------------------------------------------------------------------------------------------------------------------------

#Intersection method
def calculate_proxy_intersection_method(ticker, metadata, coefficients, prices_data, index_data, liquid_bucket, date, use_index = False):
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
        liquid_bucket_tickers = metadata[(metadata['Sector'] == liquid_bucket_sector) & (metadata['Country'] == liquid_bucket_country) & (metadata['AverageRating'] == liquid_bucket_ratings)].index.to_list()
        liquid_bucket_spread = prices_data[liquid_bucket_tickers].mean(axis=0)
        global_spread = liquid_bucket_spread


    company_bucket = f"{metadata.loc[ticker, 'Sector']}, {metadata.loc[ticker, 'Country']}, {metadata.loc[ticker, 'AverageRating']}"

    if company_bucket not in coefficients.index:
        print(f"Bucket not found for Company {ticker} ")
        return global_spread
    else:
        print(f"Bucket FOUND for Company {ticker} ")
        coefficient = coefficients.loc[company_bucket, 'Coefficient']

        proxy = coefficient + global_spread

    return proxy

#CSRA method -- Can be used for both community and non-community based methods

def calculate_proxy_csra_community(ticker,  metadata, coefficients, ticker_community, prices_data, index_data, liquid_bucket, date, use_index = False):
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
        liquid_bucket_tickers = metadata[(metadata['Sector'] == liquid_bucket_sector) & (metadata['Country'] == liquid_bucket_country) & (metadata['AverageRating'] == liquid_bucket_ratings)].index.to_list()
        liquid_bucket_spread = prices_data[liquid_bucket_tickers].mean(axis=0)
        global_spread = np.log(liquid_bucket_spread)

    company_community = f'community_{ticker_community}'
    coefficients_ticker_community = coefficients[company_community]

    # get the coefficients for the sector, country and rating
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
    metadata_with_proxies['ActualSpread'] = None
    #metadata_with_proxies['ProxyIntersection'] = None
    metadata_with_proxies['ProxyCSRA'] = None
    metadata_with_proxies['ProxyCSRACommunity'] = None

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
        # try:
        #     coefficients_intersection = calculate_coefficients_intersection_method(
        #         prices_data_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=False
        #     )
        #     proxy_intersection = calculate_proxy_intersection_method(
        #         ticker_proxy, metadata, coefficients_intersection, prices_data, index_data, liquid_bucket, date, use_index=False
        #     )
        # except Exception as e:
        #     print(f"Error calculating proxy using intersection method for {ticker_proxy}: {e}")
        #     proxy_intersection = None

        try:
            #STANDARD VERSION FOR ALL COMPANIES
            coefficients_csra = calculate_proxy_coeff_csra_community(
                prices_data_proxy_method, all_companies_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=True
            )
            proxy_csra = calculate_proxy_csra_community(
                ticker_proxy, metadata, coefficients_csra, 1, prices_data, index_data, liquid_bucket, date, use_index=True
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
                ticker_proxy, metadata, coefficients_csra_community, ticker_community, prices_data, index_data, liquid_bucket, date, use_index=True
            )
        except Exception as e:
            print(f"Error calculating proxy using CSRA community method for {ticker_proxy}: {e}")
            proxy_csra_community = None
        
        # Update the copied metadata with the results
        metadata_with_proxies.loc[metadata_with_proxies['Ticker'] == ticker_proxy, 'ActualSpread'] = actual_spread
        #metadata_with_proxies.loc[metadata_with_proxies['Ticker'] == ticker_proxy, 'ProxyIntersection'] = proxy_intersection
        metadata_with_proxies.loc[metadata_with_proxies['Ticker'] == ticker_proxy, 'ProxyCSRA'] = proxy_csra
        metadata_with_proxies.loc[metadata_with_proxies['Ticker'] == ticker_proxy, 'ProxyCSRACommunity'] = proxy_csra_community

    return metadata_with_proxies



def calculate_proxy_time_series(
    tickers, metadata, company_communities, prices_data, index_data, liquid_bucket, dates
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
    - liquid_bucket: dict, liquid bucket information (e.g., Sector, Country, Rating).
    - date_range: iterable of dates.

    Returns:
    - pd.DataFrame, time series of proxy values for the specified ticker.
    """
    results = []

    for ticker in tickers:

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
                
                # Remove the ticker from proxy calculations
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
                proxy_csra = None
                proxy_csra_community = None

                # try:
                #     coefficients_intersection = calculate_coefficients_intersection_method(
                #         prices_data_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=False
                #     )
                #     proxy_intersection = calculate_proxy_intersection_method(
                #         ticker, metadata, coefficients_intersection, prices_data, index_data, liquid_bucket, date, use_index=False
                #     )
                # except Exception as e:
                #     print(f"Error calculating intersection proxy for {ticker} on {date}: {e}")
                

                #CALCULATE PROXY FOR NORMAL CSRA METHOD
                #------------------------------------------------
                try:
                    coefficients_csra = calculate_proxy_coeff_csra_community(
                        prices_data_proxy_method, all_companies_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=True
                    )
                    proxy_csra = calculate_proxy_csra_community(
                        ticker, metadata, coefficients_csra, 1, prices_data, index_data, liquid_bucket, date, use_index=True
                    )
                except Exception as e:
                    print(f"Error calculating CSRA proxy for {ticker} on {date}: {e}")

                #CALCULATE PROXY FOR COMMUNITY CSRA METHOD
                #------------------------------------------------
                try:
                    coefficients_csra_community = calculate_proxy_coeff_csra_community(
                        prices_data_proxy_method, company_communities_proxy_method, metadata_proxy_method, index_data, liquid_bucket, date, use_index=True
                    )
                    proxy_csra_community = calculate_proxy_csra_community(
                        ticker, metadata, coefficients_csra_community, ticker_community, prices_data, index_data, liquid_bucket, date, use_index=True
                    )
                except Exception as e:
                    print(f"Error calculating CSRA community proxy for {ticker} on {date}: {e}")
                
                # Append results for this date
                results.append({
                    "Ticker": ticker,
                    "Date": date,
                    "ActualSpread": actual_spread,
                    #"ProxyIntersection": proxy_intersection,
                    "ProxyCSRA": proxy_csra,
                    "ProxyCSRACommunity": proxy_csra_community,
                })

            except Exception as e:
                print(f"Error calculating proxies for {ticker} on {date}: {e}")
        
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


    for date in dates:

        #Filter the proxy time series data by the date
        ProxyCSRA_spreads_filtered = proxy_time_series_df[proxy_time_series_df['Date'] == date]
        
        #Get the actual spreads and the proxy spreads
        actual_spreads = ProxyCSRA_spreads_filtered['ActualSpread']
        proxy_spreads_csra_normal = ProxyCSRA_spreads_filtered['ProxyCSRA']
        proxy_spreads_csra_communities = ProxyCSRA_spreads_filtered['ProxyCSRACommunity']
        
        #Calculate the RMSE
        rmse_csra_normal = calculate_rmse(actual_spreads, proxy_spreads_csra_normal)
        rmse_csra_communities = calculate_rmse(actual_spreads, proxy_spreads_csra_communities)

        rmse_csra_normal_list.append(rmse_csra_normal)
        rmse_csra_communities_list.append(rmse_csra_communities)


    return rmse_csra_normal_list, rmse_csra_communities_list


def calculate_percentage_better(rmse_csra_normal_list, rmse_csra_communities_list):

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

    print(f"Number of days where RMSE for normal CSRA method > RMSE for community CSRA method : {count}")
    print(f"Percentage of days where CSRA Communities method is better: {percentage_better}%")  


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
    plt.plot(proxy_time_series['Date'], proxy_time_series['ActualSpread'], label='Actual Spread')
    #plt.plot(proxy_time_series['Date'], proxy_time_series['ProxyIntersection'], label='Proxy Intersection')
    plt.plot(proxy_time_series['Date'], proxy_time_series['ProxyCSRA'], label='Proxy CSRA')
    plt.plot(proxy_time_series['Date'], proxy_time_series['ProxyCSRACommunity'], label='Proxy CSRA Community')

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

def plot_rmse_curves(rmse_csra_normal_list, rmse_csra_communities_list, dates):

    """
    Plots the RMSE curves for normal proxy methods and community proxy methods over the given dates.

    parameters:
    rmse_csra_normal_list: list, A list of RMSE values for CSRA Normal method.
    rmse_csra_communities_list: list, A list of RMSE values for CSRA Communities method.
    dates: list, list of dates over which the cds/price data is available (dates for cds/price data and index data should be the same).
    """

    num_days = [i for i in range(len(dates))]

    #Plot RMSE for CSRA normal and CSRA communities
    plt.plot(num_days, rmse_csra_normal_list, label='CSRA Normal')
    plt.plot(num_days, rmse_csra_communities_list, label='CSRA Communities')
    plt.xlabel('Days')
    plt.ylabel('RMSE')
    plt.title(f'RMSE for CSRA Normal and CSRA Communities over {len(dates)} days')
    plt.legend()

      


