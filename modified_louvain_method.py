import numpy as np
import pandas as pd


def create_correlation_matrix(file_path):

    # Load the standardized returns data
    df_standardized_returns = pd.read_csv(file_path)
    
    # Set the 'Date' column as the index if it's not already
    if 'Date' in df_standardized_returns.columns:
        df_standardized_returns.set_index('Date', inplace=True)

    # Calculate lambda boundaries using RMT
    T = len(df_standardized_returns)
    N = len(df_standardized_returns.columns)

    #Extract the company names
    company_names = df_standardized_returns.columns.to_list()
    
    # Calculate the correlation matrix on the returns data
    correlation_matrix = df_standardized_returns.corr()



    return correlation_matrix, T, N, company_names

def calculate_C_g(correlation_matrix,T,N):
    # Calculate the eigenvalues and eigenvectors of the correlation matrix
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
    
    lambda_plus = (1 + np.sqrt(N / T))**2
    lambda_min = (1 - np.sqrt(N / T))**2  # Not used in this code but calculated as per RMT
    
    # Obtaining eigenvalues and eigenvectors above lambda_plus
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



def calculate_modularity(C_g, communities):
    """
    Calculate modularity for a given community structure.
    C_g: The matrix accounting for the difference between C and the null model (C_g = C - (C^r + C^m))
    communities: List of lists, where each inner list represents a community (list of node indices)
    """
    # Calculate C_norm (Normalization factor: sum of all elements in C_g)
    C_norm = np.sum(C_g)
    
    modularity = 0.0
    for community in communities:
        for i in community:
            for j in community:
                modularity += C_g[i, j]  # Add the covariance between nodes i and j within the same community
    modularity /= C_norm  # Normalize by the total sum of correlations
    return modularity


def louvain_method(C_g, min_size=2, modularity_threshold=0.00001):
    """
    Perform Louvain Method for community detection adapted for correlation matrices.
    C_g: Matrix accounting for the difference between C and the null model (C_g = C - (C^r + C^m))
    min_size: Minimum size of community to stop splitting further
    modularity_threshold: Threshold for modularity improvement to decide whether to continue
    """
    # Step 1: Initialize each node as its own community
    communities = [[i] for i in range(C_g.shape[0])]
    
    # Step 2: Community detection loop
    modularity = 0
    while True:
        # Step 2a: Calculate modularity for the current community structure
        modularity_new = calculate_modularity(C_g, communities)
        
        # Step 2b: Move nodes to the best neighboring community
        for community in communities:
            # Evaluate potential movement of each node in the community
            for node in community:
                best_modularity_gain = 0
                best_community = community
                
                # Try moving the node to each of the neighboring communities
                for neighbor_community in communities:
                    if node not in neighbor_community:
                        new_communities = [c if c != community else community + [node] for c in communities]
                        modularity_gain = calculate_modularity(C_g, new_communities) - modularity_new

                        if modularity_gain > best_modularity_gain:
                            best_modularity_gain = modularity_gain
                            best_community = neighbor_community

                # Move the node to the best community if there's a gain
                if best_community != community:
                    community.remove(node)
                    best_community.append(node)

        print("Old:", modularity)
        print("New:", modularity_new)
        # Step 2c: Check if modularity improvement is below threshold
        modularity_new = calculate_modularity(C_g, communities)
        if abs(modularity_new - modularity) < modularity_threshold:
            break #Terminates while loop
        modularity = modularity_new

    # Step 3: Return the final community structure
    return communities


def aggregate_communities(C_g, communities):
    """
    Aggregate communities into supernodes (hypernodes) by summing their correlations.
    C_g: Matrix accounting for the difference between C and the null model (C_g = C - (C^r + C^m))
    communities: List of communities (each is a list of nodes)
    """

    #If there are n original communities and after aggregation, we have k supernodes, new_C_g will be a k×kk×k matrix.
    new_C_g = np.zeros((len(communities), len(communities)))

    #For each community in communities, calculate the sum of the correlation between each node in a pair of communities to get the super node correlations
    #Example, for community A and B, it will sum up the pairwise correlations of every node in A with every node in B to get the correlation between super node A and super node B
    for i, community_i in enumerate(communities):
        for j, community_j in enumerate(communities):
            if i != j:
                new_C_g[i, j] = np.sum(C_g[community_i, :][:, community_j])
            else:
                new_C_g[i, i] = np.sum(C_g[community_i, :][:, community_i])  # Self-loop (variance)
    return new_C_g

def recursive_louvain(C_g, modularity_threshold=0.00001):
    """
    Recursively apply Louvain method to find communities, then aggregate and repeat.
    C_g: Matrix accounting for the difference between C and the null model (C_g = C - (C^r + C^m))
    modularity_threshold: Threshold for modularity improvement to decide whether to continue
    """
    communities = louvain_method(C_g)  # Apply Louvain method to the current level

    # Calculate the initial modularity
    modularity_old = calculate_modularity(C_g, communities)
    
    # Apply Louvain method recursively to create a coarser network
    aggregated_C_g = aggregate_communities(C_g, communities)
    
    # Calculate the modularity after aggregation
    modularity_new = calculate_modularity(aggregated_C_g, communities)
    
    #print(modularity_new)

    # If modularity improvement is below the threshold, stop recursion
    if abs(modularity_new - modularity_old) < modularity_threshold:
        return communities  # Return the current community structure
    
    # Continue recursion on the aggregated network
    return recursive_louvain(aggregated_C_g, modularity_threshold)


if __name__ == "__main__":

    correlation_matrix,T,N,company_names = create_correlation_matrix('returns_standardized.csv')  
    C_g = calculate_C_g(correlation_matrix,T,N)

    # Extracting a submatrix (rows 1 to 2, columns 2 to 3)
    sub_matrix = C_g[0:5, 0:5]
    # Run the Louvain method on the matrix C_g
    final_communities = recursive_louvain(C_g)  
    print(final_communities)



# The final_communities will contain the hierarchical community structure at various levels
