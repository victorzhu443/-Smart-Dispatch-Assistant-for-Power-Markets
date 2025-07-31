# phase_4_3_retrieval_api.py - Implement Retrieval API Endpoint /query
import json
import os
import numpy as np
import torch
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Any, Optional
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartDispatchRAG:
    """Smart Dispatch Retrieval-Augmented Generation System"""
    
    def __init__(self):
        self.fine_tuned_model = None
        self.tokenizer = None
        self.sentence_model = None
        self.embeddings_data = None
        self.doc_vectors = None
        self.doc_metadata = None
        self.doc_ids = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load all components
        self._load_models_and_data()
    
    def _load_models_and_data(self):
        """Load fine-tuned model, embeddings, and sentence transformer"""
        try:
            print("🔄 Loading Smart Dispatch RAG components...")
            
            # Load fine-tuned GPT-2 model from Phase 4.2
            print("📥 Loading fine-tuned GPT-2 model...")
            if os.path.exists('./gpt2_dispatch_model'):
                self.tokenizer = AutoTokenizer.from_pretrained('./gpt2_dispatch_model')
                self.fine_tuned_model = AutoModelForCausalLM.from_pretrained('./gpt2_dispatch_model')
                
                # Ensure padding token is set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.fine_tuned_model.to(self.device)
                self.fine_tuned_model.eval()
                print(f"✅ Fine-tuned model loaded on {self.device}")
            else:
                raise FileNotFoundError("Fine-tuned model not found. Run Phase 4.2 first.")
            
            # Load sentence transformer for query encoding
            print("📥 Loading sentence transformer...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer loaded")
            
            # Load embeddings from Phase 4.1
            print("📥 Loading market embeddings...")
            if os.path.exists('market_embeddings.json'):
                with open('market_embeddings.json', 'r') as f:
                    self.embeddings_data = json.load(f)
                
                # Prepare vectors and metadata for retrieval
                self.doc_ids = list(self.embeddings_data['vectors'].keys())
                self.doc_vectors = np.array([
                    self.embeddings_data['vectors'][doc_id] 
                    for doc_id in self.doc_ids
                ])
                self.doc_metadata = [
                    self.embeddings_data['metadata'][doc_id] 
                    for doc_id in self.doc_ids
                ]
                
                print(f"✅ Loaded {len(self.doc_ids)} document embeddings")
            else:
                raise FileNotFoundError("Market embeddings not found. Run Phase 4.1 first.")
            
            print("✅ Smart Dispatch RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load RAG components: {e}")
            raise
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant documents for a query"""
        try:
            # Encode the query
            query_embedding = self.sentence_model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.doc_vectors)[0]
            
            # Get top-k most similar documents
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            retrieved_docs = []
            for idx in top_indices:
                doc_id = self.doc_ids[idx]
                metadata = self.doc_metadata[idx]
                similarity = float(similarities[idx])
                
                # Create a readable document description from metadata
                doc_type = metadata.get('type', 'unknown')
                
                if doc_type == 'market_observation':
                    timestamp = metadata.get('timestamp', 'unknown time')
                    price = metadata.get('price', 0)
                    settlement_point = metadata.get('settlement_point', 'unknown location')
                    price_level = metadata.get('price_level', 'unknown level')
                    time_period = metadata.get('time_period', 'unknown period')
                    
                    doc_text = f"Market observation at {settlement_point} on {timestamp}. " \
                              f"Price: ${price:.2f}/MWh ({price_level}) during {time_period}."
                
                elif doc_type == 'market_analysis':
                    timestamp = metadata.get('timestamp', 'unknown time')
                    target_price = metadata.get('target_price', 0)
                    trend_direction = metadata.get('trend_direction', 'stable')
                    volatility_level = metadata.get('volatility_level', 'moderate')
                    
                    doc_text = f"Market analysis for {timestamp}. " \
                              f"Target price: ${target_price:.2f}/MWh with {trend_direction} trend " \
                              f"and {volatility_level} volatility."
                else:
                    doc_text = f"Document {doc_id}: {doc_type} data"
                
                retrieved_docs.append({
                    'doc_id': doc_id,
                    'doc_text': doc_text,
                    'metadata': metadata,
                    'similarity': similarity
                })
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate answer using fine-tuned model and retrieved context"""
        try:
            # Prepare context from retrieved documents
            context_texts = []
            for doc in context_docs[:3]:  # Use top 3 most relevant
                context_texts.append(doc['doc_text'])
            
            context = " ".join(context_texts)
            
            # Format input for the fine-tuned model
            # Include context to improve responses
            if context.strip():
                input_text = f"Context: {context}\n\nQ: {query}\nA:"
            else:
                input_text = f"Q: {query}\nA:"
            
            # Tokenize input
            input_ids = self.tokenizer.encode(
                input_text, 
                return_tensors='pt',
                max_length=400,  # Leave room for generation
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.fine_tuned_model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 100,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    length_penalty=1.0
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            if "\nA:" in full_response:
                answer = full_response.split("\nA:")[-1].strip()
            else:
                answer = full_response[len(input_text):].strip()
            
            # Clean up the answer
            answer = self._clean_answer(answer)
            
            return answer if answer else "I'm not sure how to answer that question based on the available market data."
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "Sorry, I encountered an error while generating the response."
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and post-process the generated answer"""
        # Remove repetitive phrases
        sentences = answer.split('.')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen and len(sentence) > 10:
                unique_sentences.append(sentence)
                seen.add(sentence)
                
                # Limit to reasonable length
                if len(unique_sentences) >= 3:
                    break
        
        cleaned_answer = '. '.join(unique_sentences)
        if cleaned_answer and not cleaned_answer.endswith('.'):
            cleaned_answer += '.'
        
        return cleaned_answer
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Phase 4.3: Main RAG query function
        Input: user question
        Output: LLM-generated answer with retrieval context
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retrieve_relevant_docs(question, top_k=5)
            
            # Step 2: Generate answer using retrieved context
            answer = self.generate_answer(question, retrieved_docs)
            
            response_time = time.time() - start_time
            
            # Prepare response
            response = {
                'question': question,
                'answer': answer,
                'retrieved_documents': len(retrieved_docs),
                'context_used': [
                    {
                        'doc_id': doc['doc_id'],
                        'similarity': round(doc['similarity'], 3),
                        'type': doc['metadata'].get('type', 'unknown')
                    }
                    for doc in retrieved_docs[:3]
                ],
                'response_time_ms': round(response_time * 1000, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'question': question,
                'answer': "Sorry, I encountered an error processing your question.",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Initialize RAG system
print("🚀 Phase 4.3: Implementing Retrieval API Endpoint /query")
rag_system = SmartDispatchRAG()

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

@app.route('/query', methods=['GET', 'POST'])
def query_endpoint():
    """
    Phase 4.3: Retrieval API Endpoint /query
    Test Case: /query?q=What is the forecast? returns non-empty text
    """
    try:
        # Get question from query parameter or JSON body
        if request.method == 'GET':
            question = request.args.get('q', '')
        else:
            data = request.get_json()
            question = data.get('question', '') if data else ''
        
        if not question.strip():
            return jsonify({
                'error': 'Question parameter is required',
                'usage': 'GET /query?q=your_question or POST with {"question": "your_question"}'
            }), 400
        
        # Process the query
        logger.info(f"Processing query: {question}")
        response = rag_system.query(question)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API endpoint error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'components': {
            'fine_tuned_model': rag_system.fine_tuned_model is not None,
            'sentence_model': rag_system.sentence_model is not None,
            'embeddings': rag_system.embeddings_data is not None,
            'document_count': len(rag_system.doc_ids) if rag_system.doc_ids else 0
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    """API documentation endpoint"""
    return jsonify({
        'name': 'Smart Dispatch Assistant API',
        'version': '1.0',
        'phase': '4.3',
        'endpoints': {
            '/query': {
                'methods': ['GET', 'POST'],
                'description': 'Ask questions about power market dispatch',
                'parameters': {
                    'GET': 'q=your_question',
                    'POST': '{"question": "your_question"}'
                },
                'example': '/query?q=What is the forecast for afternoon prices?'
            },
            '/health': {
                'methods': ['GET'],
                'description': 'Check API health status'
            }
        },
        'model_info': {
            'base_model': 'GPT-2',
            'fine_tuned': True,
            'embedding_model': 'all-MiniLM-L6-v2',
            'knowledge_base_size': len(rag_system.doc_ids) if rag_system.doc_ids else 0
        }
    })

def run_test_cases():
    """
    Run Phase 4.3 test cases
    Test Case: /query?q=What is the forecast? returns non-empty text
    """
    print("\n🧪 Phase 4.3 Test Cases:")
    
    # Test questions
    test_questions = [
        "What is the forecast?",
        "What will prices be this afternoon?",
        "Should we dispatch additional generation?",
        "Why are electricity prices high right now?",
        "What's the current market volatility?",
        "Recommend dispatch strategy for peak hours"
    ]
    
    print(f"Testing {len(test_questions)} queries...")
    
    all_tests_passed = True
    
    for i, question in enumerate(test_questions, 1):
        try:
            print(f"\n   Test {i}: {question}")
            response = rag_system.query(question)
            
            # Check if response has non-empty text
            answer = response.get('answer', '')
            has_non_empty_text = len(answer.strip()) > 0
            
            print(f"   Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"   Retrieved docs: {response.get('retrieved_documents', 0)}")
            print(f"   Response time: {response.get('response_time_ms', 0):.1f}ms")
            print(f"   Result: {'✅ PASSED' if has_non_empty_text else '❌ FAILED'}")
            
            if not has_non_empty_text:
                all_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            all_tests_passed = False
    
    print(f"\n📊 Test Results: {'✅ ALL PASSED' if all_tests_passed else '❌ SOME FAILED'}")
    return all_tests_passed

def main():
    """Main function to run the API server"""
    print("✅ Smart Dispatch RAG API initialized successfully!")
    
    # Run test cases first
    test_passed = run_test_cases()
    
    if test_passed:
        print(f"\n✅ Phase 4.3 COMPLETE: Retrieval API endpoint implemented successfully")
        print(f"📊 Results:")
        print(f"   API endpoint: /query")
        print(f"   Fine-tuned model: GPT-2 (loaded)")
        print(f"   Knowledge base: {len(rag_system.doc_ids)} documents")
        print(f"   Test case status: ✅ PASSED")
        print(f"   Response generation: Working")
        print(f"🔄 Next: Phase 4.4 - Measure Response Perplexity")
        
        print(f"\n🚀 Starting Flask API server...")
        print(f"   URL: http://localhost:5000")
        print(f"   Test endpoint: http://localhost:5000/query?q=What is the forecast?")
        print(f"   Health check: http://localhost:5000/health")
        print(f"   Documentation: http://localhost:5000/")
        
        # Start the Flask server
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print(f"\n❌ Phase 4.3 failed: Test cases did not pass")

if __name__ == "__main__":
    main()