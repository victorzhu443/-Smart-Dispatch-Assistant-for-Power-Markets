# test_ercot_auth.py
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ERCOTClient:
    def __init__(self, username, password, subscription_key):
        self.username = username
        self.password = password  
        self.subscription_key = subscription_key
        self.client_id = "fec253ea-0d06-4272-a5e6-b478baeecd70"  # Fixed ERCOT client ID
        self.token_url = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
        self.base_url = "https://api.ercot.com/api/public-reports"
        self.token = None
        self.token_expires = None
    
    def get_token(self):
        """Get OAuth2 Bearer token"""
        print(f"🔧 Attempting OAuth with username: {self.username}")
        
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
        
        print(f"🔧 OAuth response status: {response.status_code}")
        
        if response.status_code == 200:
            token_data = response.json()
            self.token = token_data['id_token']
            self.token_expires = datetime.now() + timedelta(seconds=3600)
            print("✅ OAuth token obtained successfully!")
            return True
        else:
            print(f"❌ OAuth failed: {response.text}")
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
    
    def get_data_from_url(self, url):
        """Make API call to full URL (for following artifact links)"""
        headers = self.get_headers()
        
        print(f"🔧 Making API call to: {url}")
        response = requests.get(url, headers=headers)
        return response
    
    def get_data(self, endpoint):
        """Make API call to specified endpoint"""
        url = f"{self.base_url}/{endpoint}"
        return self.get_data_from_url(url)

def test_hello_world(client):
    """Test basic API access - Phase 1.2 Test Case"""
    print("\n🧪 Testing API call (Phase 1.2)...")
    
    # Settlement Point Prices endpoint
    endpoint = "np6-788-cd"  
    
    try:
        # Step 1: Get report metadata
        response = client.get_data(endpoint)
        
        if response.status_code == 200:
            metadata = response.json()
            print(f"✅ Step 1.2 PASSED: HTTP 200 - Got report metadata")
            print(f"📋 Report: {metadata['name']}")
            print(f"📝 Description: {metadata['description']}")
            
            # Step 2: Get actual data from artifacts
            if 'artifacts' in metadata:
                artifacts = metadata['artifacts']
                print(f"🔗 Found {len(artifacts)} data artifacts")
                
                if artifacts:
                    artifact = artifacts[0]
                    print(f"📊 Artifact: {artifact.get('displayName', 'No name')}")
                    
                    # Follow the endpoint link to get data schema
                    if '_links' in artifact and 'endpoint' in artifact['_links']:
                        data_url = artifact['_links']['endpoint']['href']
                        print(f"🔗 Getting data schema from: {data_url}")
                        
                        # Make call to get data schema
                        schema_response = client.get_data_from_url(data_url)
                        print(f"🔧 Schema response status: {schema_response.status_code}")
                        
                        if schema_response.status_code == 200:
                            schema_data = schema_response.json()
                            
                            print(f"✅ Got data schema")
                            print(f"📊 Report: {schema_data['report']['reportDisplayName']}")
                            
                            # Show available fields
                            fields = schema_data.get('fields', [])
                            print(f"🔑 Available fields ({len(fields)}):")
                            for field in fields:
                                print(f"   • {field['name']} ({field['dataType']}) - {field['label']}")
                            
                            # Step 3: Try to get actual data records
                            print(f"\n🔍 Attempting to get actual data records...")
                            
                            # Try different approaches to get actual data
                            attempts = [
                                f"{data_url}?size=10",  # Try with size parameter
                                f"{data_url}/records?size=10",  # Try records endpoint
                                f"{data_url}?format=json&size=10",  # Try with format
                            ]
                            
                            for i, attempt_url in enumerate(attempts, 1):
                                print(f"🔧 Attempt {i}: {attempt_url}")
                                records_response = client.get_data_from_url(attempt_url)
                                print(f"   Status: {records_response.status_code}")
                                
                                if records_response.status_code == 200:
                                    records_data = records_response.json()
                                    
                                    if isinstance(records_data, list):
                                        print(f"   ✅ SUCCESS! Got {len(records_data)} records")
                                        if records_data:
                                            sample = records_data[0]
                                            print(f"   📊 Sample record: {sample}")
                                            return records_data
                                    elif isinstance(records_data, dict) and 'data' in records_data:
                                        actual_records = records_data['data']
                                        print(f"   ✅ SUCCESS! Got {len(actual_records)} records in 'data' field")
                                        if actual_records:
                                            sample = actual_records[0]
                                            print(f"   📊 Sample record: {sample}")
                                            return actual_records
                                    else:
                                        print(f"   ⚠️ Unexpected format: {type(records_data)}")
                                        if isinstance(records_data, dict):
                                            print(f"   Keys: {list(records_data.keys())}")
                                else:
                                    error_text = records_response.text[:100] if records_response.text else "No error text"
                                    print(f"   ❌ Failed: {error_text}")
                            
                            print(f"⚠️ Could not get actual records, but schema is available")
                            return schema_data
                        else:
                            print(f"❌ Failed to get data schema: {schema_response.status_code}")
                            print(f"Response text: {schema_response.text[:200]}...")
                    else:
                        print("⚠️ No endpoint link found in artifact")
                else:
                    print("⚠️ Artifacts list is empty")
            else:
                print("⚠️ No artifacts field in metadata")
            
            return metadata
        else:
            print(f"❌ Step 1.2 FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ API call error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🚀 Starting ERCOT API Authentication Test...")
    
    # Load credentials from environment variables (CORRECT way)
    username = os.getenv('ERCOT_API_USERNAME')
    password = os.getenv('ERCOT_API_PASSWORD')
    subscription_key = os.getenv('ERCOT_API_SUBSCRIPTION_KEY')
    
    # Verify credentials were loaded
    if not all([username, password, subscription_key]):
        print("❌ Missing credentials! Check your .env file.")
        print(f"Username: {'✅' if username else '❌'}")
        print(f"Password: {'✅' if password else '❌'}")
        print(f"Subscription Key: {'✅' if subscription_key else '❌'}")
        return
    
    print(f"📧 Using username: {username}")
    print(f"🔑 Using subscription key: {subscription_key[:8]}...")
    
    # Test authentication
    client = ERCOTClient(
        username=username,
        password=password, 
        subscription_key=subscription_key
    )
    
    # Test hello world API call
    test_hello_world(client)

if __name__ == "__main__":
    main()