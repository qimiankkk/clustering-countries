import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd

COLOR_PALETTE = px.colors.qualitative.Vivid

def create_choropleth_map(df, cluster_col, title="Global Clusters"):
    df_plot = df.copy()
    df_plot['Cluster_ID'] = "Cluster " + df_plot[cluster_col].astype(str)
    
    hover_data = [col for col in df_plot.columns if col not in ['Cluster_ID', 'Cluster', cluster_col]]
    
    fig = px.choropleth(
        df_plot,
        locations=df_plot.index,
        color='Cluster_ID',
        hover_name=df_plot.index,
        hover_data=hover_data,
        color_discrete_sequence=COLOR_PALETTE,
        title=title,
        projection="natural earth"
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def create_pca_plot(df_scaled, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_scaled)
    
    plot_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    plot_df['Cluster_ID'] = "Cluster " + labels.astype(str)
    
    fig = px.scatter(
        plot_df, x='PC1', y='PC2', color='Cluster_ID',
        color_discrete_sequence=COLOR_PALETTE,
        title="PCA 2D Cluster Separation (with Centroids)"
    )
    
    # Calculate and overlay centroids
    centroids = plot_df.groupby('Cluster_ID')[['PC1', 'PC2']].mean().reset_index()
    for i, row in centroids.iterrows():
        fig.add_scatter(
            x=[row['PC1']], y=[row['PC2']],
            mode='markers',
            marker=dict(size=14, symbol='x', color='black', line=dict(width=2, color='white')),
            name=f"{row['Cluster_ID']} Centroid",
            showlegend=False,
            hoverinfo='skip'
        )
        
    return fig
