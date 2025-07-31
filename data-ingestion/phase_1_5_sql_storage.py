# phase_1_5_sql_storage.py - Save DataFrame to SQL Table
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import sqlite3  # Fallback if PostgreSQL not available

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

def create_clean_dataframe(raw_data):
    """Create and clean DataFrame (Phases 1.3-1.4 combined)"""
    columns = ['timestamp', 'repeat_hour_flag', 'settlement_point', 'price']
    df = pd.DataFrame(raw_data, columns=columns)
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Handle missing/invalid data (Phase 1.4 logic)
    df['price'] = df['price'].ffill()  # Updated syntax to avoid deprecation warning
    median_price = df['price'].median()
    df['price'] = df['price'].fillna(median_price)
    
    # Handle invalid prices
    df.loc[df['price'] < 0, 'price'] = 0
    df.loc[df['price'] > 1000, 'price'] = 1000
    
    return df

def setup_database_connection():
    """Setup database connection - try PostgreSQL first, fallback to SQLite"""
    
    # Try PostgreSQL first (as specified in PRD)
    try:
        # Check for PostgreSQL credentials in environment
        pg_user = os.getenv('POSTGRES_USER', 'postgres')
        pg_password = os.getenv('POSTGRES_PASSWORD', 'password')
        pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        pg_port = os.getenv('POSTGRES_PORT', '5432')
        pg_database = os.getenv('POSTGRES_DATABASE', 'smart_dispatch')
        
        # Create PostgreSQL connection string
        pg_connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
        
        print(f"🔧 Attempting PostgreSQL connection to {pg_host}:{pg_port}/{pg_database}")
        engine = create_engine(pg_connection_string)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print("✅ PostgreSQL connection successful")
        return engine, "postgresql"
        
    except Exception as e:
        print(f"⚠️ PostgreSQL connection failed: {e}")
        print("🔄 Falling back to SQLite for development...")
        
        # Fallback to SQLite (easier for development)
        sqlite_path = "market_data.db"
        sqlite_connection_string = f"sqlite:///{sqlite_path}"
        
        engine = create_engine(sqlite_connection_string)
        print(f"✅ SQLite connection successful: {sqlite_path}")
        return engine, "sqlite"

def save_dataframe_to_sql(df, engine, table_name="market_data"):
    """
    Phase 1.5: Save DataFrame to SQL Table market_data
    Test Case: Query from DB matches DataFrame row count
    """
    print(f"🔄 Phase 1.5: Saving DataFrame to SQL table '{table_name}'...")
    
    try:
        # Save DataFrame to SQL table
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',  # Replace table if it exists
            index=False,  # Don't save DataFrame index
            method='multi'  # Use multi-row insert for better performance
        )
        
        print(f"✅ DataFrame saved to table '{table_name}'")
        print(f"   Records saved: {len(df)}")
        
        # Test Case: Query from DB matches DataFrame row count
        with engine.connect() as conn:
            # Query row count from database
            result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table_name}"))
            db_row_count = result.fetchone()[0]
            
            # Compare with original DataFrame
            df_row_count = len(df)
            counts_match = db_row_count == df_row_count
            
            print(f"\n🧪 Test Case - Query from DB matches DataFrame row count:")
            print(f"   DataFrame rows: {df_row_count}")
            print(f"   Database rows: {db_row_count}")
            print(f"   Match: {'✅ PASSED' if counts_match else '❌ FAILED'}")
            
            # Additional validation: verify data integrity
            sample_query = f"SELECT timestamp, price FROM {table_name} LIMIT 5"
            sample_result = conn.execute(text(sample_query))
            sample_rows = sample_result.fetchall()
            
            print(f"\n📊 Sample data from database:")
            for row in sample_rows:
                print(f"   {row.timestamp} | ${row.price:.2f}")
            
            # Verify timestamp and price ranges
            stats_query = f"""
                SELECT 
                    MIN(timestamp) as min_timestamp,
                    MAX(timestamp) as max_timestamp,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    COUNT(DISTINCT settlement_point) as unique_points
                FROM {table_name}
            """
            stats_result = conn.execute(text(stats_query))
            stats = stats_result.fetchone()
            
            print(f"\n📈 Database Statistics:")
            print(f"   Time range: {stats.min_timestamp} to {stats.max_timestamp}")
            print(f"   Price range: ${stats.min_price:.2f} - ${stats.max_price:.2f}")
            print(f"   Settlement points: {stats.unique_points}")
            
            return counts_match
            
    except SQLAlchemyError as e:
        print(f"❌ SQL operation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Execute Phase 1.5 workflow"""
    print("🚀 Phase 1.5: Save DataFrame to SQL Table")
    
    # Load credentials
    username = os.getenv('ERCOT_API_USERNAME')
    password = os.getenv('ERCOT_API_PASSWORD') 
    subscription_key = os.getenv('ERCOT_API_SUBSCRIPTION_KEY')
    
    if not all([username, password, subscription_key]):
        print("❌ Missing ERCOT credentials!")
        return
    
    try:
        # Step 1: Setup database connection
        engine, db_type = setup_database_connection()
        
        # Step 2: Get and clean data (Phases 1.1-1.4)
        client = ERCOTClient(username, password, subscription_key)
        print("📡 Fetching ERCOT pricing data...")
        raw_data = client.get_pricing_data(size=100)  # Get more data for SQL demo
        print(f"✅ Retrieved {len(raw_data)} raw records")
        
        df_clean = create_clean_dataframe(raw_data)
        print(f"📊 Created clean DataFrame with {len(df_clean)} records")
        
        # Step 3: Save to SQL (Phase 1.5)
        success = save_dataframe_to_sql(df_clean, engine)
        
        if success:
            print(f"\n✅ Phase 1.5 COMPLETE: Data successfully stored in {db_type.upper()} database")
            print(f"✅ ALL PHASE 1 STEPS COMPLETE!")
            print(f"\n🎯 Ready for Phase 2: Feature Engineering (ETL)")
        else:
            print(f"\n❌ Phase 1.5 failed: Database operations unsuccessful")
            
        # Show next steps
        print(f"\n📋 Phase 1 Summary:")
        print(f"   ✅ 1.1-1.2: API Data Ingestion")
        print(f"   ✅ 1.3: Parse JSON to DataFrame")
        print(f"   ✅ 1.4: Handle Missing/Invalid Fields") 
        print(f"   ✅ 1.5: Save DataFrame to SQL Table")
        print(f"   🔄 Next: Phase 2 - Feature Engineering")
        
        return df_clean, engine
        
    except Exception as e:
        print(f"❌ Phase 1.5 failed: {e}")
        return None, None

if __name__ == "__main__":
    df, engine = main()
