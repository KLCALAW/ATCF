import networkx as nx
import matplotlib.pyplot as plt
import random

def display_communities(community_list):

    # Initialize an empty graph
    G = nx.Graph()

    # Function to generate a random color
    def random_color():
        return [random.random() for _ in range(len(community_list))]  

    # Add nodes and edges based on communities and assign random colors to each community
    community_colors = []  # Store colors for each community
    for community in community_list:
        color = random_color()  # Generate a random color for the community
        community_colors.extend([color] * len(community))  # Apply the color to all nodes in the community
        for stock in community:
            G.add_node(stock)
        for j in range(len(community)):
            for k in range(j + 1, len(community)):
                G.add_edge(community[j], community[k])  # Connect nodes within the same community

    # Draw the graph with random colors
    pos = nx.spring_layout(G)  
    nx.draw_networkx_nodes(G, pos, node_color=community_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    plt.title("Stock Communities Network with Random Colors")
    plt.axis('off')
    plt.show()
