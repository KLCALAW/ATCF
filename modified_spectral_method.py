import pandas as pd
import numpy as np

def create_correlation_matrix(data):
    """
    Create a correlation matrix from a file path or a DataFrame.
    Args:
        data (str or pd.DataFrame): Path to the CSV file or a DataFrame.
    Returns:
        tuple: Correlation matrix (as NumPy array), T, N, and company names.
    """
    if isinstance(data, str):  # If data is a file path
        df_standardized_returns = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):  # If data is already a DataFrame
        df_standardized_returns = data
    else:
        raise ValueError("Input must be a file path or a DataFrame.")
    
    # Set the 'Date' column as the index if it's not already
    if 'Date' in df_standardized_returns.columns:
        df_standardized_returns.set_index('Date', inplace=True)
    
    # Calculate lambda boundaries using RMT
    T = len(df_standardized_returns)
    N = len(df_standardized_returns.columns)

    # Extract the company names
    company_names = df_standardized_returns.columns.to_list()
    
    # Calculate the correlation matrix on the returns data
    correlation_matrix = df_standardized_returns.corr().to_numpy()  # Convert to NumPy array

    return correlation_matrix, T, N, company_names

def calculate_C_g(correlation_matrix, T, N):
    """
    Calculate the denoised correlation matrix using eigenvalue filtering.
    Ensures real values are used throughout the calculation.
    
    Args:
        correlation_matrix (np.ndarray): Input correlation matrix.
        T (int): Number of time periods.
        N (int): Number of variables.
    
    Returns:
        np.ndarray: Filtered (denoised) correlation matrix.
    """
    # Calculate the eigenvalues and eigenvectors of the correlation matrix
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
    
    # Ensure real values (handle small imaginary parts caused by numerical errors)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    lambda_plus = (1 + np.sqrt(N / T))**2
    lambda_min = (1 - np.sqrt(N / T))**2  # Not used in this code but calculated as per RMT
    
    # Obtain eigenvalues and eigenvectors above lambda_plus
    denoised_eigenvalues = []
    denoised_eigenvectors = []

    
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue > lambda_plus:
            denoised_eigenvalues.append(eigenvalue)
            denoised_eigenvectors.append(eigenvectors[:, index])  # Corresponding eigenvector
    
    # Remove the largest eigenvalue (global mode) from denoised values
    if denoised_eigenvalues:
        max_value = max(denoised_eigenvalues)
        max_index = denoised_eigenvalues.index(max_value)
        denoised_eigenvalues.pop(max_index)
        denoised_eigenvectors.pop(max_index)
    
    # Reconstruct the filtered correlation matrix C^(g)
    C_g = np.zeros_like(correlation_matrix)
    for i, eigenvalue in enumerate(denoised_eigenvalues):
        eigenvector = np.array(denoised_eigenvectors[i]).reshape(-1, 1)  # Column vector
        C_g += eigenvalue * (eigenvector @ eigenvector.T)  # Outer product
    
    # Return the filtered correlation matrix
    return C_g
    
def spectral_method(C_g):
    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(C_g)
    max_eigenvalue_index = np.argmax(eigenvalues)
    leading_eigenvector = eigenvectors[:, max_eigenvalue_index]
    
    community_1 = []
    community_2 = []
    
    # Creating the communities based on the sign of the eigenvector
    for i in range(len(leading_eigenvector)):
        if leading_eigenvector[i] > 0:
            community_1.append(i)  
        else:
            community_2.append(i) 
    
    return [community_1, community_2]

def calculate_modularity(C_g, partitions, corr_matrix):
    # C_norm is the total sum of C_g (Eq.38)
    c_norm = np.sum(np.triu(corr_matrix))

    modularity = 0.0
    
    # Calculate modularity based on the partition
    #Since the nodes have been assigned to communities, and we are summing over these communitiies seperately, there is no need for the (sisj +1)/2 term, shown in the paper
    for community in partitions:
        for i in community:
            for j in community:
                if i <= j:
                    modularity += C_g[i, j]
    
    # Normalize modularity by C_norm
    modularity /= c_norm
    return modularity

def recursive_spectral_method(C_g, corr_matrix, company_names, min_size=2, modularity_threshold=0.00001):
    result_communities = []
    modularities = []

    # Recursive function to split communities
    def split_community(community_nodes):

        # If community is too small, add it directly to result_communities
        if len(community_nodes) <= min_size:
            result_communities.append(community_nodes)
            return

        # Extract the submatrix for the current community
        submatrix = C_g[np.ix_(community_nodes, community_nodes)]

        # Apply spectral method to split into two communities
        communities = spectral_method(submatrix)

        # Map the sub-community indices back to the original indices
        community_1 = [community_nodes[i] for i in communities[0]]
        community_2 = [community_nodes[i] for i in communities[1]]
        # Calculate modularity before and after the split
        initial_modularity = calculate_modularity(C_g, [community_nodes], corr_matrix)
        new_modularity = calculate_modularity(C_g, [community_1, community_2], corr_matrix)
        # Check if the split improves modularity significantly
        if (new_modularity - initial_modularity) > modularity_threshold:
            # Recursively split each resulting community
            split_community(community_1)
            split_community(community_2)
            modularities.append(new_modularity)
        else:
            # If modularity gain is too low, add the original community without splitting
            result_communities.append(community_nodes)
    
    # Start recursive splitting from the entire set of nodes
    all_nodes = list(range(len(C_g)))


    split_community(all_nodes)

    company_communities = []

    for partition in result_communities:
        company_list = []
        for i in partition:
            company_list.append(company_names[i])
        company_communities.append(company_list)

 
    return result_communities, company_communities, modularities
    

if __name__ == "__main__":
    correlation_matrix,T,N,company_names = create_correlation_matrix('returns_standardized_S&P.csv')  
    C_g = calculate_C_g(correlation_matrix,T,N)
    result_communities, company_communities, modularities = recursive_spectral_method(C_g, correlation_matrix, company_names,min_size=2, modularity_threshold=0.00001)
    print(modularities)
    print(len(company_communities))
    print(result_communities)
    