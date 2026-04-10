import pandas as pd
import numpy as np
import os

def create_sample_data():
    os.makedirs('data', exist_ok=True)
    np.random.seed(42)
    
    dates = pd.date_range(start='2022-01-01', periods=104, freq='W')
    
    # Base trend + seasonality
    t = np.arange(104)
    trend = 10000 + 50 * t
    seasonality = 1000 * np.sin(2 * np.pi * t / 52)
    noise = np.random.normal(0, 500, 104)
    
    sales = trend + seasonality + noise
    
    # Anomaly at week 67 (index 66)
    anomaly_index = 66
    sales[anomaly_index] *= 0.6  # 40% drop
    
    # Ad spend correlated with sales
    ad_spend = sales * 0.1 + np.random.normal(0, 100, 104)
    ad_spend[anomaly_index] *= 0.5  # Drops similarly
    
    regions = ['North', 'South', 'East', 'West']
    categories = ['Electronics', 'Clothing', 'Food', 'Home']
    
    data = pd.DataFrame({
        'date': dates,
        'sales': sales.astype(int),
        'ad_spend': ad_spend.astype(int),
        'region': [regions[i % 4] for i in range(104)],
        'product_category': [categories[i % 4] for i in range(104)]
    })
    
    data.to_csv('data/sample_data.csv', index=False)
    print("Sample data generated successfully at data/sample_data.csv")

if __name__ == '__main__':
    create_sample_data()
