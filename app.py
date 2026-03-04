import streamlit as st
import pandas as pd
import numpy as np

from data_loader import fetch_all_data, preprocess_data, INDICATORS_DOC
from models import run_kmeans, custom_gmm_uniform_prior
from visualization import create_choropleth_map, create_pca_plot
from llm_insights import generate_cluster_insights

st.set_page_config(page_title="Global Demographics Clustering", layout="wide")

st.title("Global Demographic & Economic Clustering")

st.sidebar.header("Configuration")

selected_indicator_names = st.sidebar.multiselect(
    "Select Indicators to Cluster By",
    options=list(INDICATORS_DOC.keys()),
    default=list(INDICATORS_DOC.keys())
)

selected_indicators = {k: INDICATORS_DOC[k] for k in selected_indicator_names}

model_type = st.sidebar.radio("Select Model", ["K-Means", "Custom GMM (Uniform Prior)"])

n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4)

api_key = st.sidebar.text_input("Gemini API Key (optional if in ENV)", type="password")

if st.sidebar.button("Run Clustering"):
    if not selected_indicators:
        st.error("Please select at least one indicator.")
    else:
        with st.spinner("Fetching data from World Bank API..."):
            raw_data = fetch_all_data(selected_indicators)
            
        if raw_data is None or raw_data.empty:
            st.error("Failed to fetch data from World Bank. Try adjusting the indicators.")
        else:
            with st.spinner("Preprocessing Data..."):
                features = list(selected_indicators.keys())
                df_scaled, df_raw_complete, scaler = preprocess_data(raw_data, features)
                
            st.success(f"Data successfully fetched & preprocessed! Total countries analyzed: {len(df_scaled)}")
            
            with st.spinner(f"Running {model_type} Model..."):
                if model_type == "K-Means":
                    labels = run_kmeans(df_scaled.values, n_clusters)
                else:
                    labels = custom_gmm_uniform_prior(df_scaled.values, n_clusters)
                    
            df_raw_complete['Cluster'] = labels
            
            # Interactive Map
            st.subheader("Interactive Cluster Map")
            map_fig = create_choropleth_map(df_raw_complete, 'Cluster', title=f"{model_type} Clustering of the World")
            st.plotly_chart(map_fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("PCA Separation")
                pca_fig = create_pca_plot(df_scaled.values, labels)
                st.plotly_chart(pca_fig, use_container_width=True)
                
            with col2:
                st.subheader("Data Preview")
                st.dataframe(df_raw_complete)
                
            # GenAI Insights
            st.subheader("Cluster Insights from Gemini")
            with st.spinner("Generating Insights..."):
                key_to_use = api_key if api_key else None
                insights = generate_cluster_insights(df_raw_complete, 'Cluster', api_key=key_to_use)
                st.info(insights)
else:
    st.write("Configure the parameters in the sidebar and click **Run Clustering** to begin.")
