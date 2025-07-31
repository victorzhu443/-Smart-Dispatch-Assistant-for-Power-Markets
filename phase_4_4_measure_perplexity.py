# phase_4_4_measure_perplexity.py - Measure Response Perplexity
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import statistics
import math

load_dotenv()

class PerplexityEvaluator:
    """Evaluate perplexity of fine-tuned model vs base model"""
    
    def __init__(self):
        self.base_model = None
        self.base_tokenizer = None
        self.fine_tuned_model = None
        self.fine_tuned_tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load base GPT-2 and fine-tuned models"""
        print("🔄 Loading models for perplexity evaluation...")
        
        try:
            # Load base GPT-2 model
            print("📥 Loading base GPT-2 model...")
            self.base_tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.base_model = AutoModelForCausalLM.from_pretrained('gpt2')
            
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
            self.base_model.to(self.device)
            self.base_model.eval()
            print("✅ Base GPT-2 model loaded")
            
            # Load fine-tuned model
            print("📥 Loading fine-tuned GPT-2 model...")
            if os.path.exists('./gpt2_dispatch_model'):
                self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained('./gpt2_dispatch_model')
                self.fine_tuned_model = AutoModelForCausalLM.from_pretrained('./gpt2_dispatch_model')
                
                if self.fine_tuned_tokenizer.pad_token is None:
                    self.fine_tuned_tokenizer.pad_token = self.fine_tuned_tokenizer.eos_token
                
                self.fine_tuned_model.to(self.device)
                self.fine_tuned_model.eval()
                print("✅ Fine-tuned GPT-2 model loaded")
            else:
                raise FileNotFoundError("Fine-tuned model not found. Run Phase 4.2 first.")
                
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            raise
    
    def generate_reference_responses(self) -> List[Dict[str, str]]:
        """Generate GPT-quality reference responses for dispatch questions"""
        print("🔄 Generating GPT-quality reference responses...")
        
        # High-quality reference Q&A pairs (simulating GPT-4 quality responses)
        reference_responses = [
            {
                "question": "Should we dispatch the gas peaker for afternoon peak hours?",
                "reference": "Based on current market conditions showing elevated afternoon pricing typically above $45/MWh during peak demand periods (2-6 PM), dispatching the gas peaker is recommended. The unit's marginal cost of approximately $42/MWh makes it profitable during these high-price intervals. Monitor real-time LMP data and be prepared to start the unit 30 minutes before peak demand begins."
            },
            {
                "question": "What is the price forecast for tomorrow morning?",
                "reference": "Tomorrow morning prices are expected to range from $28-35/MWh during the 6-10 AM ramp-up period. The forecast shows moderate volatility with prices starting low around $28/MWh at 6 AM and gradually increasing to $35/MWh by 10 AM as demand rises. Weather conditions appear normal with no significant load stress expected."
            },
            {
                "question": "Why are electricity prices spiking right now?",
                "reference": "Current price spikes are driven by multiple factors: high cooling demand due to above-normal temperatures, unplanned outages reducing available supply by approximately 800 MW, and transmission constraints limiting imports from neighboring regions. These conditions have pushed real-time LMPs above $65/MWh, representing a significant premium over the typical $35/MWh baseline."
            },
            {
                "question": "What's the current market volatility level?",
                "reference": "Current market volatility is elevated with a coefficient of variation around 0.45, indicating high price uncertainty. This is above the normal range of 0.15-0.25, suggesting increased risk for both generation scheduling and load serving. The volatility is primarily driven by supply-demand imbalances and weather-related demand fluctuations."
            },
            {
                "question": "Should we increase generation output for evening peak?",
                "reference": "Yes, increasing generation output for the evening peak (6-9 PM) is recommended. Price forecasts show expected LMPs of $48-55/MWh during this period, well above most units' marginal costs. The evening peak typically sees 15-20% higher demand than afternoon levels, creating profitable dispatch opportunities for available capacity."
            },
            {
                "question": "What does the load forecast show for peak demand?",
                "reference": "The load forecast indicates peak demand of approximately 68,500 MW expected between 4-5 PM today. This represents a 8% increase from yesterday's peak due to higher cooling loads. The forecast shows sustained high demand through 7 PM before beginning to decline. Grid operators should prepare for potential emergency procedures if demand approaches the 70,000 MW warning level."
            },
            {
                "question": "Is the current price trend bullish or bearish?",
                "reference": "The current price trend is moderately bullish with a positive slope of 0.12 $/MWh per hour over the past 24 hours. Forward curve analysis shows prices rising from current levels of $42/MWh toward $48/MWh by evening peak. This upward trend is supported by tight supply conditions and increasing cooling demand forecasts."
            },
            {
                "question": "What's the dispatch recommendation for Unit 3?",
                "reference": "Unit 3 should be dispatched for the afternoon and evening periods. With a marginal cost of $39/MWh and current/forecasted LMPs above $44/MWh, the unit is expected to generate positive margins of $5-8/MWh. Recommend starting the unit at 1 PM to be ready for the 2 PM dispatch period and running through 9 PM when prices typically decline below economic levels."
            },
            {
                "question": "How do transmission constraints affect pricing?",
                "reference": "Current transmission constraints on the northern interface are limiting imports by 300 MW, creating a $4-6/MWh price premium in the constrained zone. These constraints typically persist during high-demand periods and can increase locational marginal prices by 10-15% compared to unconstrained conditions. Generation within the constrained area becomes more valuable during these periods."
            },
            {
                "question": "What's the grid stress level indicator?",
                "reference": "Current grid stress indicators show elevated conditions with reserve margins at 12% compared to the preferred 15% minimum. Operating reserves are adequate but tight, with spinning reserves at 1,200 MW and non-spinning at 800 MW. The system is in normal operations but operators should monitor closely for any additional outages that could trigger emergency procedures."
            }
        ]
        
        print(f"✅ Generated {len(reference_responses)} high-quality reference responses")
        return reference_responses
    
    def calculate_perplexity(self, model, tokenizer, text: str) -> float:
        """Calculate perplexity of a model on given text"""
        try:
            # Tokenize the text
            inputs = tokenizer.encode(text, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
            
        except Exception as e:
            print(f"⚠️ Error calculating perplexity: {e}")
            return float('inf')
    
    def generate_model_response(self, model, tokenizer, question: str) -> str:
        """Generate response from a model for given question"""
        try:
            input_text = f"Q: {question}\nA:"
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 80,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer part
            if "\nA:" in response:
                answer = response.split("\nA:")[-1].strip()
            else:
                answer = response[len(input_text):].strip()
            
            return answer[:200]  # Limit length for fair comparison
            
        except Exception as e:
            print(f"⚠️ Error generating response: {e}")
            return ""
    
    def evaluate_perplexity_comparison(self, reference_responses: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Phase 4.4: Measure Response Perplexity
        Test Case: Perplexity < base model baseline
        """
        print("🔄 Phase 4.4: Measuring response perplexity...")
        
        results = {
            'base_model_perplexities': [],
            'fine_tuned_perplexities': [],
            'questions': [],
            'base_responses': [],
            'fine_tuned_responses': [],
            'reference_texts': []
        }
        
        print(f"\n🧪 Test Case - Fine-tuned perplexity < base model baseline:")
        
        for i, qa_pair in enumerate(reference_responses):
            question = qa_pair['question']
            reference_text = qa_pair['reference']
            
            print(f"\n   Question {i+1}: {question[:60]}...")
            
            # Generate responses from both models
            base_response = self.generate_model_response(
                self.base_model, self.base_tokenizer, question
            )
            fine_tuned_response = self.generate_model_response(
                self.fine_tuned_model, self.fine_tuned_tokenizer, question
            )
            
            # Calculate perplexity against reference text
            base_perplexity = self.calculate_perplexity(
                self.base_model, self.base_tokenizer, reference_text
            )
            fine_tuned_perplexity = self.calculate_perplexity(
                self.fine_tuned_model, self.fine_tuned_tokenizer, reference_text
            )
            
            print(f"   Base model perplexity: {base_perplexity:.2f}")
            print(f"   Fine-tuned perplexity: {fine_tuned_perplexity:.2f}")
            print(f"   Improvement: {'✅' if fine_tuned_perplexity < base_perplexity else '❌'}")
            
            # Store results
            results['questions'].append(question)
            results['reference_texts'].append(reference_text)
            results['base_responses'].append(base_response)
            results['fine_tuned_responses'].append(fine_tuned_response)
            results['base_model_perplexities'].append(base_perplexity)
            results['fine_tuned_perplexities'].append(fine_tuned_perplexity)
        
        # Calculate overall statistics
        avg_base_perplexity = statistics.mean(results['base_model_perplexities'])
        avg_fine_tuned_perplexity = statistics.mean(results['fine_tuned_perplexities'])
        
        # Test case: Check if fine-tuned model has lower perplexity
        test_passed = avg_fine_tuned_perplexity < avg_base_perplexity
        improvement_pct = ((avg_base_perplexity - avg_fine_tuned_perplexity) / avg_base_perplexity) * 100
        
        results.update({
            'avg_base_perplexity': avg_base_perplexity,
            'avg_fine_tuned_perplexity': avg_fine_tuned_perplexity,
            'improvement_pct': improvement_pct,
            'test_passed': test_passed,
            'questions_improved': sum(1 for i in range(len(results['base_model_perplexities'])) 
                                    if results['fine_tuned_perplexities'][i] < results['base_model_perplexities'][i])
        })
        
        print(f"\n📊 Overall Results:")
        print(f"   Average base model perplexity: {avg_base_perplexity:.2f}")
        print(f"   Average fine-tuned perplexity: {avg_fine_tuned_perplexity:.2f}")
        print(f"   Improvement: {improvement_pct:.1f}%")
        print(f"   Questions improved: {results['questions_improved']}/{len(reference_responses)}")
        print(f"   Result: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        return results
    
    def create_perplexity_visualization(self, results: Dict[str, Any]):
        """Create visualization of perplexity comparison"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Perplexity comparison bar chart
            questions_short = [q[:30] + "..." for q in results['questions']]
            x_pos = np.arange(len(questions_short))
            
            axes[0, 0].bar(x_pos - 0.2, results['base_model_perplexities'], 0.4, 
                          label='Base GPT-2', color='lightcoral', alpha=0.7)
            axes[0, 0].bar(x_pos + 0.2, results['fine_tuned_perplexities'], 0.4, 
                          label='Fine-tuned GPT-2', color='lightblue', alpha=0.7)
            axes[0, 0].set_xlabel('Questions')
            axes[0, 0].set_ylabel('Perplexity')
            axes[0, 0].set_title('Perplexity Comparison by Question')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(questions_short, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Average perplexity comparison
            models = ['Base GPT-2', 'Fine-tuned GPT-2']
            avg_perplexities = [results['avg_base_perplexity'], results['avg_fine_tuned_perplexity']]
            colors = ['lightcoral', 'lightblue']
            
            bars = axes[0, 1].bar(models, avg_perplexities, color=colors, alpha=0.7)
            axes[0, 1].set_ylabel('Average Perplexity')
            axes[0, 1].set_title('Average Perplexity Comparison')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, avg_perplexities):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. Improvement histogram
            improvements = [(base - ft) for base, ft in zip(results['base_model_perplexities'], 
                                                           results['fine_tuned_perplexities'])]
            axes[1, 0].hist(improvements, bins=8, color='green', alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Perplexity Improvement')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Perplexity Improvements')
            axes[1, 0].axvline(x=0, color='red', linestyle='--', label='No improvement')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Scatter plot of base vs fine-tuned perplexity
            axes[1, 1].scatter(results['base_model_perplexities'], 
                              results['fine_tuned_perplexities'], 
                              alpha=0.7, s=60, color='purple')
            
            # Add diagonal line (y=x) for reference
            min_perp = min(min(results['base_model_perplexities']), 
                          min(results['fine_tuned_perplexities']))
            max_perp = max(max(results['base_model_perplexities']), 
                          max(results['fine_tuned_perplexities']))
            axes[1, 1].plot([min_perp, max_perp], [min_perp, max_perp], 
                           'r--', alpha=0.7, label='No improvement line')
            
            axes[1, 1].set_xlabel('Base Model Perplexity')
            axes[1, 1].set_ylabel('Fine-tuned Model Perplexity')
            axes[1, 1].set_title('Base vs Fine-tuned Perplexity')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('perplexity_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Perplexity analysis visualization saved to: perplexity_analysis.png")
            
        except Exception as e:
            print(f"⚠️ Could not create visualization: {e}")
    
    def save_detailed_results(self, results: Dict[str, Any]):
        """Save detailed results to JSON file"""
        try:
            # Prepare data for JSON serialization
            detailed_results = {
                'evaluation_timestamp': datetime.now().isoformat(),
                'test_passed': results['test_passed'],
                'summary_statistics': {
                    'avg_base_perplexity': results['avg_base_perplexity'],
                    'avg_fine_tuned_perplexity': results['avg_fine_tuned_perplexity'],
                    'improvement_pct': results['improvement_pct'],
                    'questions_improved': results['questions_improved'],
                    'total_questions': len(results['questions'])
                },
                'detailed_results': []
            }
            
            # Add detailed question-by-question results
            for i in range(len(results['questions'])):
                detailed_results['detailed_results'].append({
                    'question': results['questions'][i],
                    'reference_text': results['reference_texts'][i],
                    'base_response': results['base_responses'][i],
                    'fine_tuned_response': results['fine_tuned_responses'][i],
                    'base_perplexity': results['base_model_perplexities'][i],
                    'fine_tuned_perplexity': results['fine_tuned_perplexities'][i],
                    'improvement': results['base_model_perplexities'][i] - results['fine_tuned_perplexities'][i],
                    'improved': results['fine_tuned_perplexities'][i] < results['base_model_perplexities'][i]
                })
            
            # Save to file
            with open('perplexity_evaluation_results.json', 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            file_size = os.path.getsize('perplexity_evaluation_results.json') / 1024
            print(f"💾 Detailed results saved to: perplexity_evaluation_results.json ({file_size:.1f} KB)")
            
        except Exception as e:
            print(f"⚠️ Could not save detailed results: {e}")

def main():
    """Execute Phase 4.4 workflow"""
    print("🚀 Phase 4.4: Measure Response Perplexity")
    
    try:
        # Initialize evaluator
        evaluator = PerplexityEvaluator()
        
        # Generate high-quality reference responses
        reference_responses = evaluator.generate_reference_responses()
        
        # Evaluate perplexity comparison (Phase 4.4 main task)
        results = evaluator.evaluate_perplexity_comparison(reference_responses)
        
        # Create visualizations
        evaluator.create_perplexity_visualization(results)
        
        # Save detailed results
        evaluator.save_detailed_results(results)
        
        if results['test_passed']:
            print(f"\n✅ Phase 4.4 COMPLETE: Perplexity evaluation successful")
            print(f"📊 Results:")
            print(f"   Base model avg perplexity: {results['avg_base_perplexity']:.2f}")
            print(f"   Fine-tuned avg perplexity: {results['avg_fine_tuned_perplexity']:.2f}")
            print(f"   Improvement: {results['improvement_pct']:.1f}%")
            print(f"   Test case status: ✅ PASSED")
            print(f"   Questions improved: {results['questions_improved']}/{len(reference_responses)}")
            print(f"   Visualization: perplexity_analysis.png")
            print(f"   Detailed results: perplexity_evaluation_results.json")
            print(f"🔄 Next: Phase 5 - Backend Services (Docker/Kubernetes)")
            
            return results
        else:
            print(f"\n❌ Phase 4.4 failed: Fine-tuned model did not outperform base model")
            print(f"   This might indicate need for more training data or different hyperparameters")
            return None
            
    except Exception as e:
        print(f"❌ Phase 4.4 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()