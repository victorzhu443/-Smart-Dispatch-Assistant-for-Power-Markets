# phase_1_3_dataframe.py - Parse JSON to Pandas DataFrame
import pandas as pd
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
        # Direct call to the working endpoint
        url = f"{self.base_url}/np6-788-cd/lmp_node_zone_hub?size={size}"
        
        response = requests.get(url, headers=self.get_headers())
        
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        else:
            raise Exception(f"API call failed: {response.status_code}")

def parse_json_to_dataframe(raw_data):
    """
    Phase 1.3: Parse JSON to Pandas DataFrame
    Test Case: Print DataFrame head with expected schema
    """
    print("🔄 Phase 1.3: Converting JSON to DataFrame...")
    
    # Define column names based on ERCOT schema
    columns = ['timestamp', 'repeat_hour_flag', 'settlement_point', 'price']
    
    # Create DataFrame from array data
    df = pd.DataFrame(raw_data, columns=columns)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure price is numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    print(f"✅ DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    print(f"📊 Columns: {list(df.columns)}")
    print(f"📈 Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"⏱️ Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Test Case: Print DataFrame head with expected schema
    print(f"\n🧪 Test Case - DataFrame Head:")
    print(df.head())
    
    # Verify schema matches PRD requirements
    required_fields = ['timestamp', 'price']
    schema_check = all(field in df.columns for field in required_fields)
    print(f"\n✅ Schema Check: {schema_check} - Required fields {required_fields} present")
    
    return df

def main():
    """Execute Phase 1.3 workflow"""
    print("🚀 Phase 1.3: Parse JSON to DataFrame")
    
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
        # Step 1: Get raw pricing data (reusing our working API call)
        print("📡 Fetching ERCOT pricing data...")
        raw_data = client.get_pricing_data(size=50)  # Get 50 records for testing
        print(f"✅ Retrieved {len(raw_data)} raw records")
        
        # Step 2: Parse to DataFrame (Phase 1.3)
        df = parse_json_to_dataframe(raw_data)
        
        # Step 3: Basic validation (Phase 1.4 preview)
        print(f"\n🔍 Data Quality Check:")
        print(f"   Null timestamps: {df['timestamp'].isnull().sum()}")
        print(f"   Null prices: {df['price'].isnull().sum()}")
        print(f"   Data types: {df.dtypes.to_dict()}")
        
        print(f"\n✅ Phase 1.3 COMPLETE: DataFrame ready for Phase 1.4 (Handle Missing Data)")
        
        return df
        
    except Exception as e:
        print(f"❌ Phase 1.3 failed: {e}")
        return None

if __name__ == "__main__":
    df = main()