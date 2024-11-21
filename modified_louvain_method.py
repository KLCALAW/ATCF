

import numpy as np
from modified_spectral_method import *
import copy
import random


#TODO: Adjust change in modularity formula --- DONE ---
#TODO: Fix issue with map --- DONE ---
#TODO: Adjust logic for empty communities --- DONE ---
#TODO: Optimize algorithm
#TODO: Hypernodes 
#TODO: Check if we need to add i<=j in the modularity calculation 



def calculate_modularity_with_potential_move(node,current_index,neighbour_community_index,communities, modularity_matrix):
    
    "Calculate the modularity of a potential move of a node from its current community to a neighbour community"

    # Get the current and neighbor communities
    current_community = communities[current_index]
    neighbour_community = communities[neighbour_community_index]

    #CALCULATE MODULARITY GAIN FROM ADDING NODE TO NEW COMMUNITY
    #---------------------------------------------------

    # Calculate old modularity of the neighbor community
    # old_modularity_new_community = sum(
    #     modularity_matrix[i][j] 
    #     for i in neighbour_community
    #     for j in neighbour_community
    #     if i <= j
    # )

    # Calculate new modularity after adding the node to the neighbor community
    temp_neighbour_community = neighbour_community + [node] #We create a temp variable to avoid modifying the original community
    new_modularity_new_community = sum(  #The new modularity can be computed by adding the modularity without the node to the additional gain after adding the node
        modularity_matrix[node][j] #We only check pairwise correlations with the node we just added to the community
        #for i in temp_neighbour_community
        for j in temp_neighbour_community
        #if i <= j
    )

    #change_from_addition = new_modularity_new_community - old_modularity_new_community
    change_from_addition  = new_modularity_new_community

    #CALCULATE MODULARITY LOSS FROM REMOVING NODE FROM CURRENT COMMUNITY
    #---------------------------------------------------

    old_modularity_current_community = sum(
        modularity_matrix[i][j]
        for i in current_community
        for j in current_community
        if i <= j
    )

    temp_current_community = copy.deepcopy(current_community) #We create a temp variable to avoid modifying the original community
    temp_current_community.remove([node]) #Remove node from old community

    new_modularity_current_communtity = sum(
        modularity_matrix[i][j]
        for i in temp_current_community
        for j in temp_current_community
        if i <= j
    )


    change_from_removal = old_modularity_current_community - new_modularity_current_communtity



    # Normalize the modularity change
    c_norm = np.sum(modularity_matrix)

    modularity_change = (change_from_addition - change_from_removal) / c_norm

    return modularity_change


def phase_1(communities, modularity_matrix):

    "Randomly select a node, evaluate modularity changes, and move it to maximize modularity"
    
    nodes = [node for community in communities for node in community]  # Flatten all nodes into a single list
    community_map = {node: index for index, community in enumerate(communities) for node in community}  # Map nodes to their communities
    
    moved = True
    iteration = 0

    while moved == True:  # Continue until no moves improve modularities

        print(f"ITERATION {iteration}")
        print("-----------------")
        
        iteration += 1

        moved = False
        random.shuffle(nodes)  # Randomly shuffle the nodes
        
        for node in nodes:

            #print("COMMUNITY MAP BELOW:")
            #print(community_map)

            current_index = community_map[node]  # Find the node's current community index
            #print("Current index",current_index)
            current_community = communities[current_index]
            
            #print("Node:", node)
            #print("Current Node Communty:", current_community)
            
            max_modularity = 0
            best_community = None
            
            for j, neighbor_community in enumerate(communities):

                if not neighbor_community or neighbor_community == current_community:
                    continue  # Skip empty or identical communities
                
                modularity_change = calculate_modularity_with_potential_move(node, current_index, j, communities, modularity_matrix)
                
                if modularity_change > max_modularity:
                    max_modularity = modularity_change
                    best_community = neighbor_community
            
            # If a modularity improvement is found, move the node
            if max_modularity > 0 and best_community is not None:
                current_community.remove(node)
                best_community.append(node)
                community_map[node] = communities.index(best_community)  # Update the community map
                moved = True  # Indicate that a move occurred
                #print("MOVED NODE", node, "TO COMMUNITY", communities.index(best_community))

                #print("UPDATED COMMUNITY MAP:")
                #print(community_map)

        # Remove empty communities
        communities = [community for community in communities if community]

        #Update community map with new communities
        community_map = {node: index for index, community in enumerate(communities) for node in community}  # Map nodes to their communities
        
        print(f"End of iteration {iteration}, Communities: {communities},")
    
    return communities


def modified_louvain(modularity_matrix):

    #Get node indices from matrix
    node_indices = np.arange(modularity_matrix.shape[0])

    #Create initial communities
    communities = [[node] for node in node_indices]


    # print("Modularity matrix")
    # print(modularity_matrix)

    #Calculate initial modularity
    new_communities = phase_1(communities, modularity_matrix)

    return new_communities
        


