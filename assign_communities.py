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

def assign_ticker_to_community_bayes(ticker, metadata, company_communities):

    metadata = metadata.set_index('Ticker')

    ticker_sector = metadata.loc[ticker, 'Sector']
    ticker_country = metadata.loc[ticker, 'Country']
    ticker_rating = metadata.loc[ticker, 'AverageRating']

    # remove the ticker from the metadata
    metadata = metadata.drop(ticker)

    communities = copy.deepcopy(company_communities)

    # remove the ticker from the communities
    for i, community in enumerate(communities):
        if ticker in community:
            communities[i].remove(ticker)
            break

    number_of_communities = len(communities)

    if number_of_communities == 1:
        return 1

    # Calculate the total number of companies
    total_companies = sum(len(companies) for companies in communities)

    # Calculate prior probabilities for each community
    community_prior = np.array([len(companies) / total_companies for companies in communities])

    # Calculate the likelihoods of the ticker
    likelihood_ratings = np.zeros(number_of_communities)
    for i in range(number_of_communities):
        # count percentage of companies in the community with the same rating as the ticker
        likelihood_ratings[i] = sum(metadata.loc[companies, 'AverageRating'] == ticker_rating for companies in communities[i]) / len(communities[i])
    likelihood_sectors = np.zeros(number_of_communities)

    for i in range(number_of_communities):
        # count percentage of companies in the community with the same sector as the ticker
        likelihood_sectors[i] = sum(metadata.loc[companies, 'Sector'] == ticker_sector for companies in communities[i]) / len(communities[i])
    
    likelihood_countries = np.zeros(number_of_communities)
    for i in range(number_of_communities):
        # count percentage of companies in the community with the same country as the ticker
        likelihood_countries[i] = sum(metadata.loc[companies, 'Country'] == ticker_country for companies in communities[i]) / len(communities[i])

    community_posterior = community_prior * likelihood_ratings * likelihood_sectors * likelihood_countries

    if sum(community_posterior) == 0:
        # If the posterior probabilities are all zero, return the prior probabilities
        return community_prior, np.argmax(community_prior) + 1
    
    # normalize the posterior probabilities
    community_posterior = community_posterior / sum(community_posterior)

    return community_posterior, np.argmax(community_posterior) + 1

def get_original_community(ticker_proxy, company_communities):
    """
    Function to find the original community of a ticker.

    Parameters:
        ticker_proxy (str): The ticker symbol to search for.
        company_communities (list of lists): A list of communities with tickers.

    Returns:
        int: The community number where the ticker resides.
    """
    for i, community in enumerate(company_communities):
        if ticker_proxy in community:
            return i + 1
    return None  # Return None if the ticker is not found in any community

def evaluate_placement_accuracy_bayes(metadata, company_communities, criteria=None):
    """
    Evaluate the accuracy of community placement based on naives bayes classifier.

    Parameters:
        metadata (DataFrame): The metadata DataFrame containing company information.
        company_communities (list of lists): A list of communities with tickers.

    Returns:
        float: Accuracy percentage of placements.
    """
    correct_placements = 0
    incorrect_placements = 0

    for ticker_proxy in metadata['Ticker']:
        try:
            # Determine the predicted community
            array, predicted_community = assign_ticker_to_community_bayes(ticker_proxy, metadata, company_communities)

            # Get the original community
            original_community = get_original_community(ticker_proxy, company_communities)

            # Compare the two
            if predicted_community == original_community:
                correct_placements += 1
            else:
                incorrect_placements += 1
        except Exception as e:
            print(f"Error processing ticker {ticker_proxy}: {e}")

    # Calculate accuracy
    total = correct_placements + incorrect_placements
    accuracy = (correct_placements / total) * 100 if total > 0 else 0
    print(f"Correct placements: {correct_placements}")
    print(f"Incorrect placements: {incorrect_placements}")
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":

    #Prices data
    #-----------------------------------------
    prices_data = pd.read_csv('data/reshaped_data.csv')
    prices_data['Date'] = pd.to_datetime(prices_data['Date'], infer_datetime_format=True)
    prices_data = prices_data.set_index('Date')

    #Index data
    #-----------------------------------------
    index_data = pd.read_csv('ITRAXX-Europe Timeseries 20241127.csv') #To be used for b0
    index_data.rename(columns={'AsOf':'Date'}, inplace=True)
    try:
        index_data['Date'] = pd.to_datetime(index_data['Date'], format='%d-%b-%y')
    except Exception as e:
        index_data['Date'] = pd.to_datetime(index_data['Date'], format='%d/%b/%y')
    index_data = index_data.sort_values(by='Date', ascending=True)
    #Metadata
    #-----------------------------------------
    # metadata = pd.read_csv('data/metadata_batched_ratings.csv')
    metadata = pd.read_csv('data/metadata.csv')


    #Community detection
    #-----------------------------------------
    correlation_matrix,T,N,company_names = create_correlation_matrix('data/eur_data_standardized_returns2.csv')
    C_g = calculate_C_g(correlation_matrix, T, N)
    result_communities, company_communities, modularities = recursive_spectral_method(C_g, correlation_matrix, company_names, min_size=2, modularity_threshold=0.00001)

    ticker = 'ZINCO'

    # Find the original community
    
    original_community = get_original_community(ticker, company_communities)

    print(original_community)

    probabilites, assigned_community = assign_ticker_to_community_bayes(ticker, metadata, company_communities)

    print(probabilites, assigned_community)

    accuracy = evaluate_placement_accuracy_bayes(metadata, company_communities)