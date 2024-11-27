import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
from modified_spectral_method import recursive_spectral_method, create_correlation_matrix, calculate_C_g

# Function to plot pie charts by user-selected criteria (sector, region, or country) using Plotly
def plot_group_distribution(groups, df, criteria='Sector'):
    
    unique_sectors = df['Sector'].unique()
    unique_regions = df['Region'].unique()
    unique_countries = df['Country'].unique()

    # Define a more distinguishable color palette
    color_palette = pc.qualitative.Plotly + pc.qualitative.Dark24 + pc.qualitative.Light24
    sector_colors = dict(zip(unique_sectors, color_palette[:len(unique_sectors)]))
    region_colors = dict(zip(unique_regions, color_palette[:len(unique_regions)]))
    country_colors = dict(zip(unique_countries, color_palette[:len(unique_countries)]))

    if criteria not in ['Sector', 'Region', 'Country']:
        raise ValueError("Invalid criteria! Use 'Sector', 'Region', or 'Country'.")

    color_dict_sector = {'Sector': sector_colors, 'Region': region_colors, 'Country': country_colors}
    color_dict = color_dict_sector[criteria]
    
    # Setting up the subplot grid
    num_groups = len(groups)
    ncols = 3  # Maximum of 3 columns
    nrows = (num_groups + ncols - 1) // ncols  # Calculate required rows based on number of groups
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[f'Group {i+1}' for i in range(num_groups)], specs=[[{'type': 'domain'}]*ncols]*nrows)
    
    # Add each group's pie chart to the subplots
    for idx, tickers in enumerate(groups):
        row = idx // ncols + 1
        col = idx % ncols + 1

        # Filter data for the current group
        group_data = df[df['Ticker'].isin(tickers)]

        # Count the occurrences of each sector, region, or country
        counts = group_data[criteria].value_counts()
        
        # Prepare colors for the pie chart
        colors = [color_dict[key] for key in counts.index]

        # Add the pie chart to the subplot
        fig.add_trace(go.Pie(labels=counts.index, values=counts, marker=dict(colors=colors), showlegend=False), row=row, col=col)

    # Create a legend manually at the bottom of the figure
    fig.update_layout(
        title_text=f"{criteria} Distribution for Groups",
        showlegend=True,
        legend=dict(
            title=criteria,
            orientation="h",
            yanchor="bottom",
            y=-0.3,  # Adjust to move legend up or down
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        )
    )

    # Add dummy traces to populate the shared legend with colors
    for label, color in color_dict.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=color), name=label))

    fig.show()


if __name__ == '__main__':
    df = pd.read_csv('metadata.csv')
    correlation_matrix, T, N, company_names = create_correlation_matrix('eur_data_standardized_returns.csv')  
    C_g = calculate_C_g(correlation_matrix, T, N)
    result_communities, company_communities, modularities = recursive_spectral_method(C_g, correlation_matrix, company_names, min_size=2, modularity_threshold=0.00001)

    plot_group_distribution(company_communities, df, criteria='Country')
