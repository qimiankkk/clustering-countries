from data_loader import fetch_all_data, preprocess_data, INDICATORS_DOC
from models import run_kmeans, custom_gmm_uniform_prior

def test_pipeline():
    print("Testing data fetch...")
    indicators = {'GDP': INDICATORS_DOC['GDP per capita (current US$)'], 'Population Growth': INDICATORS_DOC['Population growth (annual %)']}
    raw_data = fetch_all_data(indicators, lookback_years=5)
    
    assert raw_data is not None and not raw_data.empty, "Data fetching failed"
    print(f"Data fetched successfully for {len(raw_data)} rows.")
    
    print("Testing preprocessing...")
    df_scaled, df_raw, scaler = preprocess_data(raw_data, list(indicators.keys()))
    assert len(df_scaled) > 0, "Preprocessing resulted in empty dataframe"
    print(f"Preprocessing returned {len(df_scaled)} rows.")
    
    print("Testing K-Means...")
    labels_kmeans = run_kmeans(df_scaled.values, n_clusters=3)
    assert len(labels_kmeans) == len(df_scaled), "K-Means failed"
    print("K-Means succeeded.")
    
    print("Testing Custom GMM...")
    labels_gmm = custom_gmm_uniform_prior(df_scaled.values, n_clusters=3, max_iters=10)
    assert len(labels_gmm) == len(df_scaled), "Custom GMM failed"
    print("Custom GMM succeeded.")
    
    print("All backend tests passed!")

if __name__ == "__main__":
    test_pipeline()
