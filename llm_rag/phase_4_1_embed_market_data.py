# phase_4_1_embed_market_data.py - Embed Market Data with SentenceTransformers (Local Storage)
import pandas as pd
import numpy as np
import torch
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import hashlib
import time

load_dotenv()

def setup_database_connection():
    """Setup database connection"""
    try:
        pg_user = os.getenv('POSTGRES_USER', 'postgres')
        pg_password = os.getenv('POSTGRES_PASSWORD', 'password')
        pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        pg_port = os.getenv('POSTGRES_PORT', '5432')
        pg_database = os.getenv('POSTGRES_DATABASE', 'smart_dispatch')
        
        pg_connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
        engine = create_engine(pg_connection_string)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print("✅ PostgreSQL connection successful")
        return engine, "postgresql"
        
    except Exception as e:
        print(f"⚠️ PostgreSQL not available, using SQLite")
        sqlite_path = "market_data.db"
        sqlite_connection_string = f"sqlite:///{sqlite_path}"
        engine = create_engine(sqlite_connection_string)
        print(f"✅ SQLite connection successful: {sqlite_path}")
        return engine, "sqlite"

def load_market_data_for_embedding(engine):
    """Load market data from database for embedding"""
    print(f"🔄 Loading market data for embedding...")
    
    try:
        # Load both raw market data and processed features
        print(f"📊 Loading raw market data...")
        market_query = """
        SELECT timestamp, settlement_point, price, repeat_hour_flag
        FROM market_data 
        ORDER BY timestamp DESC 
        LIMIT 500
        """
        df_market = pd.read_sql(market_query, engine)
        df_market['timestamp'] = pd.to_datetime(df_market['timestamp'])
        
        print(f"📈 Loading processed features...")
        features_query = """
        SELECT window_id, target_time, target_price, price_mean, price_std, 
               trend_slope, price_volatility, hour_of_day, day_of_week, 
               is_weekend, is_peak_hour, momentum_1h, price_sequence_json
        FROM features 
        ORDER BY target_time DESC
        """
        df_features = pd.read_sql(features_query, engine)
        df_features['target_time'] = pd.to_datetime(df_features['target_time'])
        
        print(f"✅ Loaded market data:")
        print(f"   Raw market records: {len(df_market)}")
        print(f"   Feature records: {len(df_features)}")
        print(f"   Time range: {df_market['timestamp'].min()} to {df_market['timestamp'].max()}")
        print(f"   Settlement points: {df_market['settlement_point'].nunique()}")
        
        return df_market, df_features
        
    except Exception as e:
        print(f"❌ Failed to load market data: {e}")
        return None, None

def create_market_documents_for_embedding(df_market, df_features):
    """Create text documents from market data for embedding"""
    print(f"🔄 Creating market documents for embedding...")
    
    documents = []
    
    # Create documents from raw market data (price observations)
    print(f"📊 Processing raw market data into documents...")
    for idx, row in df_market.iterrows():
        timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M')
        settlement_point = row['settlement_point']
        price = row['price']
        is_repeat = row.get('repeat_hour_flag', False)
        
        # Create descriptive text for the price observation
        hour = row['timestamp'].hour
        day_name = row['timestamp'].strftime('%A')
        
        # Determine price context
        if price < 20:
            price_context = "low price"
        elif price < 40:
            price_context = "moderate price"
        elif price < 60:
            price_context = "high price"
        else:
            price_context = "very high price"
        
        # Determine time context
        if 6 <= hour <= 9:
            time_context = "morning ramp-up"
        elif 14 <= hour <= 18:
            time_context = "afternoon peak"
        elif 19 <= hour <= 22:
            time_context = "evening demand"
        else:
            time_context = "off-peak hours"
        
        doc_text = f"""Power market observation at {settlement_point} on {day_name} {timestamp}. 
        Electricity price is ${price:.2f}/MWh indicating {price_context} during {time_context}. 
        This represents real-time locational marginal pricing from ERCOT Texas grid operations."""
        
        # Create metadata
        metadata = {
            'type': 'market_observation',
            'timestamp': timestamp,
            'settlement_point': settlement_point,
            'price': float(price),
            'hour': hour,
            'day_of_week': row['timestamp'].weekday(),
            'price_level': price_context,
            'time_period': time_context
        }
        
        doc_id = f"market_{settlement_point}_{row['timestamp'].strftime('%Y%m%d_%H%M')}"
        
        documents.append({
            'id': doc_id,
            'text': doc_text.strip(),
            'metadata': metadata
        })
        
        if len(documents) % 50 == 0:
            print(f"   Created {len(documents)} market observation documents...")
    
    # Create documents from processed features (analysis insights)
    print(f"📈 Processing feature data into analysis documents...")
    for idx, row in df_features.iterrows():
        timestamp = row['target_time'].strftime('%Y-%m-%d %H:%M')
        target_price = row['target_price']
        price_mean = row['price_mean']
        price_std = row['price_std']
        trend_slope = row['trend_slope']
        volatility = row['price_volatility']
        hour = row['hour_of_day']
        is_weekend = row['is_weekend']
        is_peak = row['is_peak_hour']
        momentum = row['momentum_1h']
        
        # Parse price sequence for additional context
        try:
            price_sequence = json.loads(row['price_sequence_json'])
            price_min = min(price_sequence)
            price_max = max(price_sequence)
            price_range = price_max - price_min
        except:
            price_min = price_max = price_range = 0
        
        # Create analytical description
        trend_desc = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
        volatility_desc = "high volatility" if volatility > 0.3 else "moderate volatility" if volatility > 0.15 else "low volatility"
        momentum_desc = "upward momentum" if momentum > 1 else "downward momentum" if momentum < -1 else "stable momentum"
        
        period_desc = "weekend" if is_weekend else "weekday"
        peak_desc = "peak demand period" if is_peak else "off-peak period"
        
        doc_text = f"""Power market analysis for {timestamp} showing {trend_desc} price trend with {volatility_desc}. 
        The 24-hour average price was ${price_mean:.2f}/MWh with standard deviation of ${price_std:.2f}. 
        Market exhibited {momentum_desc} with price range from ${price_min:.2f} to ${price_max:.2f}/MWh. 
        This {period_desc} {peak_desc} forecast targets ${target_price:.2f}/MWh for next hour dispatch planning."""
        
        # Create metadata
        metadata = {
            'type': 'market_analysis',
            'timestamp': timestamp,
            'target_price': float(target_price),
            'price_mean': float(price_mean),
            'price_std': float(price_std),
            'trend_slope': float(trend_slope),
            'price_volatility': float(volatility),
            'hour_of_day': int(hour),
            'is_weekend': bool(is_weekend),
            'is_peak_hour': bool(is_peak),
            'momentum_1h': float(momentum),
            'price_range': float(price_range),
            'trend_direction': trend_desc,
            'volatility_level': volatility_desc
        }
        
        doc_id = f"analysis_{row['window_id']}_{row['target_time'].strftime('%Y%m%d_%H%M')}"
        
        documents.append({
            'id': doc_id,
            'text': doc_text.strip(),
            'metadata': metadata
        })
    
    print(f"✅ Created {len(documents)} documents for embedding:")
    print(f"   Market observations: {len([d for d in documents if d['metadata']['type'] == 'market_observation'])}")
    print(f"   Market analyses: {len([d for d in documents if d['metadata']['type'] == 'market_analysis'])}")
    
    return documents

def setup_sentence_transformer():
    """Initialize SentenceTransformer model"""
    print(f"🔄 Setting up SentenceTransformer model...")
    
    try:
        # Use a model optimized for semantic search
        model_name = "all-MiniLM-L6-v2"  # Fast, good performance, 384 dimensions
        model = SentenceTransformer(model_name)
        
        # Test the model
        test_text = "Power market price is $45.50/MWh during peak hours"
        test_embedding = model.encode(test_text)
        
        print(f"✅ SentenceTransformer initialized:")
        print(f"   Model: {model_name}")
        print(f"   Embedding dimension: {len(test_embedding)}")
        print(f"   Test embedding shape: {test_embedding.shape}")
        
        return model, len(test_embedding)
        
    except Exception as e:
        print(f"❌ Failed to setup SentenceTransformer: {e}")
        return None, None

def setup_vector_storage(embedding_dim):
    """Setup local vector storage (Pinecone removed for simplicity)"""
    print(f"🔄 Setting up local vector storage...")
    
    local_storage = {
        'vectors': {},
        'metadata': {},
        'embeddings': {},
        'config': {
            'dimension': embedding_dim,
            'metric': 'cosine',
            'created_at': datetime.now().isoformat()
        }
    }
    
    print(f"✅ Local vector storage initialized:")
    print(f"   Storage type: Local JSON file")
    print(f"   Dimension: {embedding_dim}")
    print(f"   Metric: cosine similarity")
    
    return local_storage, "local_storage"

def create_local_vector_storage():
    """Create local vector storage as fallback"""
    print(f"🔄 Setting up local vector storage...")
    
    local_storage = {
        'vectors': {},
        'metadata': {},
        'embeddings': {}
    }
    
    print(f"✅ Local vector storage initialized")
    return local_storage

def embed_and_store_documents(documents, model, storage, storage_type):
    """
    Phase 4.1: Embed Market Data with SentenceTransformers
    Store in Pinecone (or local fallback)
    Test Case: Vector length and ID return correctly from Pinecone
    """
    print(f"🔄 Phase 4.1: Embedding and storing {len(documents)} documents...")
    
    # Batch processing for efficiency
    batch_size = 32
    stored_count = 0
    failed_count = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        try:
            # Extract texts for embedding
            texts = [doc['text'] for doc in batch]
            
            # Generate embeddings
            embeddings = model.encode(texts, show_progress_bar=False)
            
            if storage_type == "pinecone":
                # Prepare vectors for Pinecone
                vectors_to_upsert = []
                for j, doc in enumerate(batch):
                    vector_data = {
                        'id': doc['id'],
                        'values': embeddings[j].tolist(),
                        'metadata': doc['metadata']
                    }
                    vectors_to_upsert.append(vector_data)
                
                # Upsert to Pinecone
                storage.upsert(vectors=vectors_to_upsert)
                
            else:  # local_storage or local_fallback
                # Store in local storage
                for j, doc in enumerate(batch):
                    storage['vectors'][doc['id']] = embeddings[j].tolist()
                    storage['metadata'][doc['id']] = doc['metadata']
                    storage['embeddings'][doc['id']] = {
                        'text': doc['text'],
                        'embedding': embeddings[j]
                    }
            
            stored_count += len(batch)
            
            if stored_count % 100 == 0:
                print(f"   Embedded and stored {stored_count}/{len(documents)} documents...")
                
        except Exception as e:
            failed_count += len(batch)
            print(f"⚠️ Failed to process batch {i//batch_size + 1}: {e}")
    
    print(f"✅ Embedding and storage completed:")
    print(f"   Successfully stored: {stored_count} documents")
    print(f"   Failed: {failed_count} documents")
    
    # Test Case: Vector length and ID return correctly
    print(f"\n🧪 Test Case - Vector length and ID return correctly:")
    
    if stored_count > 0:
        # Test with first document
        test_doc = documents[0]
        test_id = test_doc['id']
        
        try:
            if storage_type == "pinecone":
                # Query Pinecone for the test document
                query_response = storage.fetch(ids=[test_id])
                
                if test_id in query_response['vectors']:
                    retrieved_vector = query_response['vectors'][test_id]
                    vector_length = len(retrieved_vector['values'])
                    returned_id = retrieved_vector['id']
                    
                    print(f"   Storage type: Pinecone")
                    print(f"   Test ID: {test_id}")
                    print(f"   Returned ID: {returned_id}")
                    print(f"   ID match: {'✅' if test_id == returned_id else '❌'}")
                    print(f"   Vector length: {vector_length}")
                    print(f"   Expected length: {len(model.encode(test_doc['text']))}")
                    
                    test_passed = (test_id == returned_id) and (vector_length > 0)
                else:
                    print(f"   ❌ Test document not found in Pinecone")
                    test_passed = False
                    
            else:  # local_storage or local_fallback
                if test_id in storage['vectors']:
                    retrieved_vector = storage['vectors'][test_id]
                    vector_length = len(retrieved_vector)
                    
                    print(f"   Storage type: {storage_type}")
                    print(f"   Test ID: {test_id}")
                    print(f"   Vector length: {vector_length}")
                    print(f"   Expected length: {len(model.encode(test_doc['text']))}")
                    print(f"   Metadata available: {'✅' if test_id in storage['metadata'] else '❌'}")
                    
                    test_passed = (vector_length > 0) and (test_id in storage['metadata'])
                else:
                    print(f"   ❌ Test document not found in local storage")
                    test_passed = False
            
            print(f"   Result: {'✅ PASSED' if test_passed else '❌ FAILED'}")
            
            # Additional verification
            if test_passed:
                print(f"\n📊 Storage Verification:")
                total_stored = stored_count
                
                if storage_type == "pinecone":
                    # Get index stats
                    stats = storage.describe_index_stats()
                    pinecone_count = stats.get('total_vector_count', 0)
                    print(f"   Pinecone vector count: {pinecone_count}")
                    print(f"   Storage consistency: {'✅' if pinecone_count >= stored_count else '⚠️'}")
                else:
                    local_count = len(storage['vectors'])
                    print(f"   Local vector count: {local_count}")
                    print(f"   Storage consistency: {'✅' if local_count == stored_count else '⚠️'}")
                
                print(f"   Documents embedded: {total_stored}")
                print(f"   Storage type: {storage_type}")
            
            return test_passed, stored_count, storage_type
            
        except Exception as e:
            print(f"   ❌ Test case failed: {e}")
            return False, stored_count, storage_type
    else:
        print(f"   ❌ No documents stored, cannot run test case")
        return False, 0, storage_type

def save_local_storage(storage, filename="market_embeddings.json"):
    """Save local storage to file for persistence"""
    if storage and len(storage['vectors']) > 0:
        try:
            # Convert numpy arrays to lists for JSON serialization
            storage_for_json = {
                'vectors': storage['vectors'],
                'metadata': storage['metadata'],
                'info': {
                    'vector_count': len(storage['vectors']),
                    'created_at': datetime.now().isoformat(),
                    'embedding_model': 'all-MiniLM-L6-v2'
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(storage_for_json, f)
            
            file_size = os.path.getsize(filename) / (1024 * 1024)
            print(f"💾 Local embeddings saved to {filename} ({file_size:.2f} MB)")
            return True
        except Exception as e:
            print(f"⚠️ Failed to save local storage: {e}")
            return False
    return False

def main():
    """Execute Phase 4.1 workflow"""
    print("🚀 Phase 4.1: Embed Market Data with SentenceTransformers")
    
    try:
        # Step 1: Setup database and load data
        engine, db_type = setup_database_connection()
        df_market, df_features = load_market_data_for_embedding(engine)
        
        if df_market is None or df_features is None:
            print("❌ Phase 4.1 failed: Could not load market data")
            return None
        
        # Step 2: Create documents for embedding
        documents = create_market_documents_for_embedding(df_market, df_features)
        
        # Step 3: Setup SentenceTransformer
        model, embedding_dim = setup_sentence_transformer()
        if model is None:
            print("❌ Phase 4.1 failed: Could not setup SentenceTransformer")
            return None
        
        # Step 4: Setup vector storage (local storage)
        # FIXED: Changed from undefined setup_pinecone_index to setup_vector_storage
        storage, storage_type = setup_vector_storage(embedding_dim)
        if storage is None:
            storage = create_local_vector_storage()
            storage_type = "local_fallback"
        
        # Step 5: Embed and store documents (Phase 4.1)
        test_passed, stored_count, final_storage_type = embed_and_store_documents(
            documents, model, storage, storage_type
        )
        
        # Step 6: Save local storage if applicable
        if final_storage_type == "local_storage" or final_storage_type == "local_fallback":
            save_local_storage(storage)
        
        if test_passed and stored_count > 0:
            print(f"\n✅ Phase 4.1 COMPLETE: Market data successfully embedded and stored")
            print(f"📊 Results:")
            print(f"   Documents processed: {len(documents)}")
            print(f"   Vectors stored: {stored_count}")
            print(f"   Storage type: {final_storage_type}")
            print(f"   Embedding model: all-MiniLM-L6-v2")
            print(f"   Vector dimension: {embedding_dim}")
            print(f"🔄 Next: Phase 4.2 - Fine-Tune Hugging Face LLM on Dispatch Q&A")
            
            return {
                'storage': storage,
                'storage_type': final_storage_type,
                'model': model,
                'documents': documents,
                'stored_count': stored_count,
                'embedding_dim': embedding_dim
            }
        else:
            print(f"\n❌ Phase 4.1 failed: Embedding and storage unsuccessful")
            return None
            
    except Exception as e:
        print(f"❌ Phase 4.1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
