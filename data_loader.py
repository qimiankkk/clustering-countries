import pandas as pd
import numpy as np
import requests
import datetime
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

INDICATORS_DOC = {
    'GDP per capita (current US$)': 'NY.GDP.PCAP.CD',
    'Life expectancy at birth, total (years)': 'SP.DYN.LE00.IN',
    'Population growth (annual %)': 'SP.POP.GROW',
    'Literacy rate, adult total (% of people ages 15 and above)': 'SE.ADT.LITR.ZS',
    'Unemployment, total (% of total labor force)': 'SL.UEM.TOTL.ZS',
    'CO2 emissions (metric tons per capita)': 'EN.ATM.CO2E.PC',
    'Inflation, consumer prices (annual %)': 'FP.CPI.TOTL.ZG',
    'Foreign direct investment, net inflows (% of GDP)': 'BX.KLT.DINV.WD.GD.ZS',
    'Access to electricity (% of population)': 'EG.ELC.ACCS.ZS'
}

def fetch_indicator_data(indicator_code, start_year, end_year):
    url = f"http://api.worldbank.org/v2/country/all/indicator/{indicator_code}?format=json&per_page=20000&date={start_year}:{end_year}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return pd.DataFrame()
    data = resp.json()
    if len(data) < 2:
        return pd.DataFrame()
    
    records = data[1]
    df = pd.DataFrame(records)
    if 'countryiso3code' not in df.columns or 'value' not in df.columns:
        return pd.DataFrame()
        
    df = df[['countryiso3code', 'date', 'value']].dropna(subset=['countryiso3code'])
    df = df[df['countryiso3code'] != '']
    return df

def fetch_all_data(indicators_dict, lookback_years=5):
    current_year = datetime.datetime.now().year
    start_year = current_year - lookback_years
    end_year = current_year
    
    merged_df = None
    
    for name, code in indicators_dict.items():
        df = fetch_indicator_data(code, start_year, end_year)
        if df.empty:
            continue
            
        df_valid = df.dropna(subset=['value'])
        df_valid = df_valid.sort_values('date', ascending=False)
        df_recent = df_valid.drop_duplicates(subset=['countryiso3code'], keep='first')
        
        df_recent = df_recent[['countryiso3code', 'value']].rename(columns={'value': name})
        
        if merged_df is None:
            merged_df = df_recent
        else:
            merged_df = pd.merge(merged_df, df_recent, on='countryiso3code', how='outer')
            
    if merged_df is not None:
        merged_df = merged_df.set_index('countryiso3code')
        
    return merged_df

def preprocess_data(df, indicators, max_missing_pct=0.3):
    raw_df = df.copy()

    # Drop countries with too many missing values
    missing_pct = df[indicators].isnull().mean(axis=1)
    df = df[missing_pct <= max_missing_pct]
    raw_df = raw_df.loc[df.index].copy()
    
    if df.empty:
        return df, raw_df, None
        
    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(df[indicators])
    df_imputed = pd.DataFrame(imputed_data, columns=indicators, index=df.index)
    
    # Update raw_df with imputed values so we have complete raw values
    raw_df.update(df_imputed)
    
    # Log Transformation for skewed variables
    df_scaled = df_imputed.copy()
    for col in indicators:
        # Check skewness, apply log1p if skewed and all strictly positive
        skewness = df_scaled[col].skew()
        if skewness > 1 and df_scaled[col].min() >= 0:
            df_scaled[col] = np.log1p(df_scaled[col])
            
    # Standard Scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_scaled)
    df_scaled = pd.DataFrame(scaled_data, columns=indicators, index=df.index)
    
    return df_scaled, raw_df, scaler
