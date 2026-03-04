import os
from google import genai
import pandas as pd

def generate_cluster_insights(raw_df, cluster_col='Cluster', api_key=None):
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
        
    if not api_key:
        return "⚠️ Google API Key is missing. Please set the GOOGLE_API_KEY environment variable or enter it in the app sidebar."
        
    client = genai.Client(api_key=api_key)
    
    # Calculate medians for each cluster
    cluster_profiles = raw_df.groupby(cluster_col).median()
    
    prompt = "You are an expert geopolitical and economic analyst.\n"
    prompt += "I have clustered the countries of the world into distinct groups based on the following median characteristics:\n\n"
    
    for cluster_id, row in cluster_profiles.iterrows():
        prompt += f"### Cluster {cluster_id}\n"
        for indicator, value in row.items():
            prompt += f"- {indicator}: {value:.2f}\n"
        prompt += "\n"
        
    prompt += "Please provide a succinct, natural-language executive summary outlining the geopolitical and economic characteristics of each cluster. "
    prompt += "Assign a short descriptive title to each cluster, and briefly explain what unites the countries within it. Format your response clearly in Markdown."
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error analyzing clusters: {str(e)}"
