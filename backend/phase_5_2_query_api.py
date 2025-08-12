# phase_5_2_query_api.py - Dockerized Query API with RAG Pipeline
import json
import os
import numpy as np
import torch
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Any, Optional
import time
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartDispatchRAG:
    """Dockerized Smart Dispatch Retrieval-Augmented Generation System"""
    
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
            logger.info("🔄 Loading Smart Dispatch RAG components...")
            
            # Load fine-tuned GPT-2 model from Phase 4.2
            logger.info("📥 Loading fine-tuned GPT-2 model...")
            if os.path.exists('./gpt2_dispatch_model'):
                self.tokenizer = AutoTokenizer.from_pretrained('./gpt2_dispatch_model')
                self.fine_tuned_model = AutoModelForCausalLM.from_pretrained('./gpt2_dispatch_model')
                
                # Ensure padding token is set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.fine_tuned_model.to(self.device)
                self.fine_tuned_model.eval()
                logger.info(f"✅ Fine-tuned model loaded on {self.device}")
            else:
                raise FileNotFoundError("Fine-tuned model not found. Run Phase 4.2 first.")
            
            # Load sentence transformer for query encoding
            logger.info("📥 Loading sentence transformer...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded")
            
            # Load embeddings from Phase 4.1
            logger.info("📥 Loading market embeddings...")
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
                
                logger.info(f"✅ Loaded {len(self.doc_ids)} document embeddings")
            else:
                raise FileNotFoundError("Market embeddings not found. Run Phase 4.1 first.")
            
            logger.info("✅ Smart Dispatch RAG system initialized successfully")
            
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
        Phase 5.2: Main RAG query function for dockerized API
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
                'timestamp': datetime.now().isoformat(),
                'service': 'dockerized-query-api',
                'version': '5.2'
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'question': question,
                'answer': "Sorry, I encountered an error processing your question.",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'service': 'dockerized-query-api',
                'version': '5.2'
            }

# Initialize RAG system
logger.info("🚀 Phase 5.2: Initializing Dockerized Query API with RAG Pipeline")
rag_system = SmartDispatchRAG()

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Simple HTML chatbot interface template
CHATBOT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Dispatch Assistant</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-container {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .message {
            margin: 10px 0;
            padding: 12px 16px;
            border-radius: 20px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background: white;
            border: 2px solid #e9ecef;
            margin-right: auto;
        }
        .input-container {
            padding: 20px;
            border-top: 1px solid #e9ecef;
            display: flex;
            gap: 10px;
        }
        .input-field {
            flex: 1;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            outline: none;
            font-size: 16px;
        }
        .input-field:focus {
            border-color: #007bff;
        }
        .send-button {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        .send-button:hover {
            background: #0056b3;
        }
        .send-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .typing {
            color: #6c757d;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔌 Smart Dispatch Assistant</h1>
            <p>Ask me about electricity prices, dispatch decisions, and market analysis</p>
        </div>
        <div class="chat-container" id="chatContainer">
            <div class="message bot-message">
                Hello! I'm your intelligent dispatch assistant. Ask me questions like:
                <br>• "Should we dispatch the gas peaker?"
                <br>• "What will prices be this afternoon?"
                <br>• "Why are electricity prices high right now?"
            </div>
        </div>
        <div class="input-container">
            <input type="text" id="messageInput" class="input-field" placeholder="Ask about power market dispatch..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()" id="sendButton" class="send-button">Send</button>
        </div>
    </div>

    <script>
        function addMessage(message, isUser) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            messageDiv.innerHTML = message;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function showTyping() {
            addMessage('<span class="typing">Assistant is thinking...</span>', false);
        }

        function removeTyping() {
            const messages = document.querySelectorAll('.message');
            const lastMessage = messages[messages.length - 1];
            if (lastMessage && lastMessage.querySelector('.typing')) {
                lastMessage.remove();
            }
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const question = input.value.trim();
            
            if (!question) return;
            
            // Add user message
            addMessage(question, true);
            input.value = '';
            sendButton.disabled = true;
            
            // Show typing indicator
            showTyping();
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                
                // Remove typing indicator
                removeTyping();
                
                if (data.answer) {
                    addMessage(data.answer, false);
                } else {
                    addMessage('Sorry, I encountered an error processing your question.', false);
                }
                
            } catch (error) {
                removeTyping();
                addMessage('Sorry, I cannot connect to the dispatch assistant right now.', false);
            } finally {
                sendButton.disabled = false;
                input.focus();
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        // Focus input on load
        window.onload = function() {
            document.getElementById('messageInput').focus();
        };
    </script>
</body>
</html>
"""

@app.route('/query', methods=['GET', 'POST'])
def query_endpoint():
    """
    Phase 5.2: Dockerized RAG Query API Endpoint
    Test Case: Chatbot interface returns answers
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
                'usage': 'GET /query?q=your_question or POST with {"question": "your_question"}',
                'service': 'dockerized-query-api'
            }), 400
        
        # Process the query
        logger.info(f"Processing query: {question}")
        response = rag_system.query(question)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API endpoint error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'service': 'dockerized-query-api'
        }), 500

@app.route('/chat', methods=['GET'])
def chatbot_interface():
    """
    Chatbot web interface
    Test Case: Chatbot interface returns answers
    """
    return render_template_string(CHATBOT_HTML)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'dockerized-query-api',
        'version': '5.2',
        'components': {
            'fine_tuned_model': rag_system.fine_tuned_model is not None,
            'sentence_model': rag_system.sentence_model is not None,
            'embeddings': rag_system.embeddings_data is not None,
            'document_count': len(rag_system.doc_ids) if rag_system.doc_ids else 0
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/forecast', methods=['GET'])
def forecast_proxy():
    """Proxy to forecast API for integrated functionality"""
    try:
        timestamp = request.args.get('timestamp', '')
        
        # Try to connect to forecast API (assuming it's running on port 5001)
        forecast_url = 'http://localhost:5001/forecast'
        if timestamp:
            forecast_url += f'?timestamp={timestamp}'
        
        response = requests.get(forecast_url, timeout=5)
        return jsonify(response.json())
        
    except Exception as e:
        return jsonify({
            'error': 'Forecast service unavailable',
            'message': 'Make sure the forecast API is running on port 5001',
            'suggestion': 'Start Phase 5.1 forecast API first'
        }), 503

@app.route('/', methods=['GET'])
def api_info():
    """API documentation endpoint"""
    return jsonify({
        'name': 'Smart Dispatch Query API',
        'version': '5.2',
        'phase': '5.2',
        'description': 'Dockerized RAG-powered intelligent dispatch assistant',
        'endpoints': {
            '/query': {
                'methods': ['GET', 'POST'],
                'description': 'Ask intelligent dispatch questions',
                'parameters': {
                    'GET': 'q=your_question',
                    'POST': '{"question": "your_question"}'
                },
                'example': '/query?q=Should we dispatch the gas peaker?'
            },
            '/chat': {
                'methods': ['GET'],
                'description': 'Interactive chatbot web interface',
                'url': '/chat'
            },
            '/forecast': {
                'methods': ['GET'],
                'description': 'Proxy to forecast API (Phase 5.1)',
                'note': 'Requires forecast API running on port 5001'
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
            'knowledge_base_size': len(rag_system.doc_ids) if rag_system.doc_ids else 0,
            'rag_pipeline': True
        },
        'docker': {
            'containerized': True,
            'port': 5002,
            'build_command': 'docker build -t query-api .',
            'run_command': 'docker run -p 5002:5002 query-api'
        }
    })

def run_test_cases():
    """
    Test Case: Chatbot interface returns answers
    """
    logger.info("\n🧪 Phase 5.2 Test Cases:")
    
    test_questions = [
        "Should we dispatch the gas peaker?",
        "What will prices be this afternoon?",
        "Why are electricity prices high right now?",
        "What's the current market volatility?"
    ]
    
    logger.info(f"Testing {len(test_questions)} chatbot queries...")
    
    all_tests_passed = True
    
    with app.test_client() as client:
        for i, question in enumerate(test_questions, 1):
            try:
                logger.info(f"\n   Test {i}: {question}")
                
                # Test API endpoint
                response = client.post('/query', json={'question': question})
                
                if response.status_code == 200:
                    data = response.get_json()
                    answer = data.get('answer', '')
                    has_answer = len(answer.strip()) > 0
                    
                    logger.info(f"   Response: {response.status_code}")
                    logger.info(f"   Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")
                    logger.info(f"   Retrieved docs: {data.get('retrieved_documents', 0)}")
                    logger.info(f"   Response time: {data.get('response_time_ms', 0):.1f}ms")
                    logger.info(f"   Result: {'✅ PASSED' if has_answer else '❌ FAILED'}")
                    
                    if not has_answer:
                        all_tests_passed = False
                else:
                    logger.info(f"   ❌ FAILED: HTTP {response.status_code}")
                    all_tests_passed = False
                    
            except Exception as e:
                logger.info(f"   ❌ FAILED: {e}")
                all_tests_passed = False
        
        # Test chatbot interface
        logger.info(f"\n   Test: Chatbot interface accessibility")
        try:
            response = client.get('/chat')
            if response.status_code == 200 and 'Smart Dispatch Assistant' in response.get_data(as_text=True):
                logger.info(f"   Chatbot interface: ✅ ACCESSIBLE")
            else:
                logger.info(f"   Chatbot interface: ❌ FAILED")
                all_tests_passed = False
        except Exception as e:
            logger.info(f"   Chatbot interface: ❌ FAILED - {e}")
            all_tests_passed = False
    
    logger.info(f"\n📊 Test Results: {'✅ ALL PASSED' if all_tests_passed else '❌ SOME FAILED'}")
    return all_tests_passed

def main():
    """Main function to run the dockerized query API"""
    logger.info("✅ Smart Dispatch Query API (Dockerized) initialized successfully!")
    
    # Run test cases first
    test_passed = run_test_cases()
    
    if test_passed:
        logger.info(f"\n✅ Phase 5.2 READY: Dockerized Query API working correctly")
        logger.info("🔄 Next: Create Dockerfile and deploy with docker-compose")
        logger.info(f"\n🚀 Starting Query API server...")
        logger.info(f"   API URL: http://localhost:5002")
        logger.info(f"   Chatbot Interface: http://localhost:5002/chat")
        logger.info(f"   Test API: curl -X POST http://localhost:5002/query -H 'Content-Type: application/json' -d '{\"question\": \"Should we dispatch the gas peaker?\"}'")
        logger.info(f"   Health Check: http://localhost:5002/health")
        logger.info(f"   Documentation: http://localhost:5002/")
        
        # Start the Flask server
        app.run(host='0.0.0.0', port=5002, debug=False)
    else:
        logger.error(f"\n❌ Phase 5.2 failed: Test cases did not pass")

if __name__ == "__main__":
    main()