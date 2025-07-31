# phase_1_4_missing_data.py - Handle Missing/Invalid Fields
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class ERCOTClient:
    def __init__(self, username, password, subscription_key):
        self.username = username
        self.password = password  
        self.subscription_key = subscription_key
        self.client_id = "fec253ea-0d06-4272-a5e6-b478baeecd70"
        self.token_url = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
        self.base_url = "https://api.ercot.com/api/public-reports"
        self.token = None
        self.token_expires = None
    
    def get_token(self):
        """Get OAuth2 Bearer token"""
        data = {
            'username': self.username,
            'password': self.password,
            'grant_type': 'password',
            'scope': 'openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access',
            'client_id': self.client_id,
            'response_type': 'id_token'
        }
        
        response = requests.post(
            self.token_url,
            data=data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        
        if response.status_code == 200:
            token_data = response.json()
            self.token = token_data['id_token']
            self.token_expires = datetime.now() + timedelta(seconds=3600)
            return True
        return False
    
    def get_headers(self):
        """Get headers with both auth methods"""
        if not self.token or datetime.now() >= self.token_expires:
            if not self.get_token():
                raise Exception("Failed to get OAuth token")
        
        return {
            "Authorization": f"Bearer {self.token}",
            "Ocp-Apim-Subscription-Key": self.subscription_key
        }
    
    def get_pricing_data(self, size=100):
        """Get ERCOT pricing data and return as clean records"""
        url = f"{self.base_url}/np6-788-cd/lmp_node_zone_hub?size={size}"
        
        response = requests.get(url, headers=self.get_headers())
        
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        else:
            raise Exception(f"API call failed: {response.status_code}")

def create_dataframe_with_issues(raw_data):
    """Create DataFrame and simulate some data quality issues for testing"""
    columns = ['timestamp', 'repeat_hour_flag', 'settlement_point', 'price']
    df = pd.DataFrame(raw_data, columns=columns)
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Simulate some data quality issues for Phase 1.4 testing
    if len(df) > 5:
        # Introduce some NaN prices
        df.loc[2, 'price'] = np.nan
        df.loc[7, 'price'] = np.nan
        
        # Introduce invalid price (negative)
        df.loc[4, 'price'] = -999.0
        
        # Introduce extreme outlier
        df.loc[6, 'price'] = 50000.0
    
    return df

def handle_missing_invalid_fields(df):
    """
    Phase 1.4: Handle Missing/Invalid Fields
    Test Case: No NaNs in critical columns after processing
    """
    print("🔄 Phase 1.4: Handling Missing/Invalid Fields...")
    
    # Initial data quality assessment
    print(f"📊 Initial Data Quality:")
    print(f"   Total records: {len(df)}")
    print(f"   Null timestamps: {df['timestamp'].isnull().sum()}")
    print(f"   Null prices: {df['price'].isnull().sum()}")
    print(f"   Invalid prices (< 0): {(df['price'] < 0).sum()}")
    print(f"   Extreme prices (> 1000): {(df['price'] > 1000).sum()}")
    
    # Store original count for comparison
    original_count = len(df)
    
    # 1. Drop rows with missing timestamps (critical field)
    df_clean = df.dropna(subset=['timestamp']).copy()
    dropped_timestamp = original_count - len(df_clean)
    if dropped_timestamp > 0:
        print(f"🗑️ Dropped {dropped_timestamp} records with missing timestamps")
    
    # 2. Handle missing prices
    missing_prices = df_clean['price'].isnull().sum()
    if missing_prices > 0:
        print(f"🔧 Found {missing_prices} missing prices")
        
        # Option 1: Drop rows with missing prices (for critical data)
        # df_clean = df_clean.dropna(subset=['price'])
        
        # Option 2: Forward fill missing prices (better for time series)
        df_clean['price'] = df_clean['price'].fillna(method='ffill')
        
        # Option 3: Use median for remaining NaNs
        median_price = df_clean['price'].median()
        df_clean['price'] = df_clean['price'].fillna(median_price)
        
        print(f"✅ Filled missing prices using forward fill + median")
    
    # 3. Handle invalid prices (negative values)
    invalid_negative = (df_clean['price'] < 0).sum()
    if invalid_negative > 0:
        print(f"🔧 Found {invalid_negative} negative prices")
        # Replace negative prices with 0 (or median)
        df_clean.loc[df_clean['price'] < 0, 'price'] = 0
        print(f"✅ Set negative prices to 0")
    
    # 4. Handle extreme outliers (likely data errors)
    # Define reasonable bounds for electricity prices ($/MWh)
    price_lower_bound = 0
    price_upper_bound = 1000  # $1000/MWh is very high but possible during scarcity
    
    extreme_high = (df_clean['price'] > price_upper_bound).sum()
    if extreme_high > 0:
        print(f"🔧 Found {extreme_high} extreme high prices (> ${price_upper_bound})")
        # Cap extreme prices at upper bound
        df_clean.loc[df_clean['price'] > price_upper_bound, 'price'] = price_upper_bound
        print(f"✅ Capped extreme prices at ${price_upper_bound}")
    
    # 5. Ensure no NaNs remain in critical columns
    critical_columns = ['timestamp', 'price']
    for col in critical_columns:
        remaining_nulls = df_clean[col].isnull().sum()
        if remaining_nulls > 0:
            print(f"⚠️ Warning: {remaining_nulls} NaNs still present in {col}")
            # Last resort: drop these rows
            df_clean = df_clean.dropna(subset=[col])
            print(f"🗑️ Dropped remaining rows with null {col}")
    
    # Final validation
    final_count = len(df_clean)
    records_lost = original_count - final_count
    
    print(f"\n✅ Phase 1.4 Data Cleaning Complete:")
    print(f"   Original records: {original_count}")
    print(f"   Clean records: {final_count}")
    print(f"   Records lost: {records_lost} ({records_lost/original_count*100:.1f}%)")
    
    # Test Case: No NaNs in critical columns after processing
    timestamp_nulls = df_clean['timestamp'].isnull().sum()
    price_nulls = df_clean['price'].isnull().sum()
    
    test_passed = (timestamp_nulls == 0) and (price_nulls == 0)
    print(f"\n🧪 Test Case - No NaNs in critical columns: {'✅ PASSED' if test_passed else '❌ FAILED'}")
    print(f"   Timestamp NaNs: {timestamp_nulls}")
    print(f"   Price NaNs: {price_nulls}")
    
    # Show final data quality
    print(f"\n📊 Final Data Quality:")
    print(f"   Price range: ${df_clean['price'].min():.2f} - ${df_clean['price'].max():.2f}")
    print(f"   Data types: {df_clean.dtypes.to_dict()}")
    
    return df_clean

def main():
    """Execute Phase 1.4 workflow"""
    print("🚀 Phase 1.4: Handle Missing/Invalid Fields")
    
    # Load credentials
    username = os.getenv('ERCOT_API_USERNAME')
    password = os.getenv('ERCOT_API_PASSWORD') 
    subscription_key = os.getenv('ERCOT_API_SUBSCRIPTION_KEY')
    
    if not all([username, password, subscription_key]):
        print("❌ Missing credentials!")
        return
    
    # Initialize client
    client = ERCOTClient(username, password, subscription_key)
    
    try:
        # Step 1: Get raw data
        print("📡 Fetching ERCOT pricing data...")
        raw_data = client.get_pricing_data(size=50)
        print(f"✅ Retrieved {len(raw_data)} raw records")
        
        # Step 2: Create DataFrame with simulated issues
        df = create_dataframe_with_issues(raw_data)
        print(f"📊 Created DataFrame with {len(df)} records (including simulated issues)")
        
        # Step 3: Handle missing/invalid fields (Phase 1.4)
        df_clean = handle_missing_invalid_fields(df)
        
        print(f"\n✅ Phase 1.4 COMPLETE: Clean DataFrame ready for Phase 1.5 (Save to SQL)")
        
        # Preview for Phase 1.5
        print(f"\n🔍 Sample of clean data:")
        print(df_clean.head())
        
        return df_clean
        
    except Exception as e:
        print(f"❌ Phase 1.4 failed: {e}")
        return None

if __name__ == "__main__":
    df_clean = main()
