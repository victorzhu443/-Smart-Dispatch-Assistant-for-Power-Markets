# phase_5_2_minimal.py - Minimal Query API (No Pandas Dependencies)
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartDispatchRAG:
    """Minimal Smart Dispatch RAG System"""
    
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
        """Main RAG query function"""
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
                'service': 'minimal-query-api',
                'version': '5.2-minimal'
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'question': question,
                'answer': "Sorry, I encountered an error processing your question.",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'service': 'minimal-query-api',
                'version': '5.2-minimal'
            }

# Initialize RAG system
logger.info("🚀 Phase 5.2: Initializing Minimal Query API")
rag_system = SmartDispatchRAG()

# Create Flask app
app = Flask(__name__)
CORS(app)

# Simple HTML chatbot interface
CHATBOT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Dispatch Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f2f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .chat-box { height: 400px; border: 1px solid #ddd; border-radius: 5px; padding: 15px; overflow-y: auto; background: #fafafa; margin-bottom: 20px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
        .user { background: #007bff; color: white; text-align: right; }
        .bot { background: #e9ecef; color: #333; }
        .input-area { display: flex; gap: 10px; }
        .input-field { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .send-btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .send-btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔌 Smart Dispatch Assistant</h1>
            <p>Ask me about electricity dispatch decisions!</p>
        </div>
        <div class="chat-box" id="chatBox">
            <div class="message bot">Hello! Ask me dispatch questions like "Should we dispatch the gas peaker?" or "What will prices be this afternoon?"</div>
        </div>
        <div class="input-area">
            <input type="text" id="messageInput" class="input-field" placeholder="Ask about dispatch..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()" class="send-btn">Send</button>
        </div>
    </div>
    
    <script>
        function addMessage(text, isUser) {
            const chatBox = document.getElementById('chatBox');
            const msg = document.createElement('div');
            msg.className = 'message ' + (isUser ? 'user' : 'bot');
            msg.textContent = text;
            chatBox.appendChild(msg);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const question = input.value.trim();
            if (!question) return;
            
            addMessage(question, true);
            input.value = '';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                const data = await response.json();
                addMessage(data.answer || 'Sorry, no response available.', false);
            } catch (error) {
                addMessage('Error: Could not connect to API.', false);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/query', methods=['GET', 'POST'])
def query_endpoint():
    """Phase 5.2: Query API Endpoint"""
    try:
        if request.method == 'GET':
            question = request.args.get('q', '')
        else:
            data = request.get_json()
            question = data.get('question', '') if data else ''
        
        if not question.strip():
            return jsonify({'error': 'Question parameter is required'}), 400
        
        logger.info(f"Processing query: {question}")
        response = rag_system.query(question)
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/chat', methods=['GET'])
def chatbot_interface():
    """Chatbot web interface"""
    return render_template_string(CHATBOT_HTML)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'minimal-query-api',
        'version': '5.2-minimal',
        'components': {
            'fine_tuned_model': rag_system.fine_tuned_model is not None,
            'sentence_model': rag_system.sentence_model is not None,
            'embeddings': rag_system.embeddings_data is not None,
            'document_count': len(rag_system.doc_ids) if rag_system.doc_ids else 0
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def api_info():
    """API documentation"""
    return jsonify({
        'name': 'Smart Dispatch Query API (Minimal)',
        'version': '5.2-minimal',
        'description': 'Simplified RAG-powered dispatch assistant',
        'endpoints': {
            '/query': 'Ask dispatch questions',
            '/chat': 'Web chatbot interface',
            '/health': 'Health check'
        },
        'test_commands': {
            'api_test': 'curl -X POST http://localhost:5002/query -H "Content-Type: application/json" -d "{\\"question\\": \\"Should we dispatch the gas peaker?\\"}";',
            'web_interface': 'http://localhost:5002/chat'
        }
    })

def main():
    """Main function"""
    logger.info("✅ Minimal Query API initialized!")
    logger.info("🧪 Testing basic functionality...")
    
    # Quick test
    try:
        test_response = rag_system.query("Should we dispatch the gas peaker?")
        if test_response.get('answer'):
            logger.info("✅ RAG system working!")
            logger.info(f"   Test answer: {test_response['answer'][:50]}...")
        else:
            logger.warning("⚠️ RAG system partially working")
    except Exception as e:
        logger.error(f"❌ RAG system error: {e}")
    
    logger.info("🚀 Starting server on http://localhost:5002")
    logger.info("   Chatbot: http://localhost:5002/chat")
    logger.info("   Health: http://localhost:5002/health")
    
    app.run(host='0.0.0.0', port=5002, debug=False)

if __name__ == "__main__":
    main()