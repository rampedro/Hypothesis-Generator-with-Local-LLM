#!/usr/bin/env python3
"""
Simplified Hypothesis Generation and Knowledge Base Relationship Visualizer
Uses Ollama local LLM and HuggingFace RoBERTa API for knowledge exploration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import logging
from typing import Dict, List, Any, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class HypothesisGenerator:
    def __init__(self):
        """Initialize the hypothesis generator with Ollama and HuggingFace APIs."""
        self.ollama_url = "http://localhost:11434"
        self.hf_api_url = "https://api-inference.huggingface.co/models/FacebookAI/roberta-large-mnli"
        self.ollama_model = "qwen2.5:0.5b"  # Small Qwen model
        
        # Test Ollama connection
        self.test_ollama_connection()
    
    def test_ollama_connection(self):
        """Test if Ollama is running and has the required model."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.ollama_model not in model_names:
                    logger.warning(f"Model {self.ollama_model} not found. Available models: {model_names}")
                    logger.info(f"Please run: ollama pull {self.ollama_model}")
                else:
                    logger.info(f"Ollama connected successfully with model {self.ollama_model}")
            else:
                logger.error("Ollama is not responding")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.info("Please ensure Ollama is running: ollama serve")
    
    def generate_hypothesis(self, premise: str, context: str = "") -> Dict[str, Any]:
        """
        Generate hypotheses based on a premise using Ollama.
        
        Args:
            premise: The main statement or premise
            context: Additional context for hypothesis generation
            
        Returns:
            Dictionary containing generated hypotheses and metadata
        """
        prompt = f"""
        Given the premise: "{premise}"
        {f"Context: {context}" if context else ""}
        
        Generate 3-5 hypotheses that could be tested or explored based on this premise.
        For each hypothesis, provide:
        1. The hypothesis statement
        2. Whether it supports, contradicts, or is neutral to the premise
        3. What evidence would be needed to test it
        
        Format your response as JSON:
        {{
            "hypotheses": [
                {{
                    "statement": "hypothesis text",
                    "relationship": "supports|contradicts|neutral",
                    "evidence_needed": "what evidence is required",
                    "confidence": "high|medium|low"
                }}
            ]
        }}
        
        JSON Response:
        """
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", 
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                })
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if json_match:
                    try:
                        hypotheses_data = json.loads(json_match.group())
                        return {
                            'success': True,
                            'premise': premise,
                            'context': context,
                            'hypotheses': hypotheses_data.get('hypotheses', []),
                            'raw_response': generated_text
                        }
                    except json.JSONDecodeError:
                        pass
                
                # Fallback: parse text response manually
                return {
                    'success': True,
                    'premise': premise,
                    'context': context,
                    'hypotheses': self._parse_text_hypotheses(generated_text),
                    'raw_response': generated_text
                }
            else:
                return {'success': False, 'error': f"Ollama API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parse_text_hypotheses(self, text: str) -> List[Dict]:
        """Parse hypotheses from text when JSON parsing fails."""
        hypotheses = []
        lines = text.split('\n')
        
        current_hypothesis = {}
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                if current_hypothesis:
                    hypotheses.append(current_hypothesis)
                current_hypothesis = {
                    'statement': line,
                    'relationship': 'neutral',
                    'evidence_needed': 'Further investigation required',
                    'confidence': 'medium'
                }
            elif line and current_hypothesis:
                current_hypothesis['statement'] += ' ' + line
        
        if current_hypothesis:
            hypotheses.append(current_hypothesis)
        
        return hypotheses[:5]  # Limit to 5 hypotheses
    
    def test_nli_relationship(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Test the relationship between premise and hypothesis using RoBERTa MNLI.
        
        Args:
            premise: The premise statement
            hypothesis: The hypothesis to test
            
        Returns:
            NLI classification results
        """
        try:
            # HuggingFace Inference API payload
            payload = {
                "inputs": {
                    "text": premise,
                    "text_pair": hypothesis
                }
            }
            
            # Note: For production use, you should add HF API token
            # headers = {"Authorization": f"Bearer {hf_token}"}
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(self.hf_api_url, 
                headers=headers, 
                json=payload,
                timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse the classification results
                if isinstance(result, list) and len(result) > 0:
                    scores = result[0]
                    
                    # Find the highest scoring label
                    max_score = 0
                    predicted_label = "NEUTRAL"
                    
                    for item in scores:
                        if item['score'] > max_score:
                            max_score = item['score']
                            predicted_label = item['label']
                    
                    return {
                        'success': True,
                        'premise': premise,
                        'hypothesis': hypothesis,
                        'predicted_label': predicted_label,
                        'confidence': max_score,
                        'all_scores': scores
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Unexpected response format from HuggingFace API'
                    }
            else:
                return {
                    'success': False,
                    'error': f"HuggingFace API error: {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            logger.error(f"Error testing NLI relationship: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_knowledge_graph(self, premise: str, hypotheses: List[Dict]) -> Dict[str, Any]:
        """
        Generate a knowledge graph structure for D3.js visualization.
        
        Args:
            premise: The main premise
            hypotheses: List of generated hypotheses
            
        Returns:
            Graph structure with nodes and links
        """
        nodes = []
        links = []
        
        # Add premise node (central node)
        nodes.append({
            'id': 'premise',
            'label': premise[:50] + '...' if len(premise) > 50 else premise,
            'type': 'premise',
            'full_text': premise,
            'weight': 3
        })
        
        # Add hypothesis nodes and links
        for i, hyp in enumerate(hypotheses):
            hyp_id = f"hypothesis_{i}"
            
            nodes.append({
                'id': hyp_id,
                'label': hyp['statement'][:40] + '...' if len(hyp['statement']) > 40 else hyp['statement'],
                'type': 'hypothesis',
                'full_text': hyp['statement'],
                'relationship': hyp.get('relationship', 'neutral'),
                'confidence': hyp.get('confidence', 'medium'),
                'evidence_needed': hyp.get('evidence_needed', ''),
                'weight': 2
            })
            
            # Add link from premise to hypothesis
            link_type = hyp.get('relationship', 'neutral')
            links.append({
                'source': 'premise',
                'target': hyp_id,
                'predicate': link_type,
                'type': 'hypothesis_relation',
                'weight': 1 if link_type == 'neutral' else 2
            })
        
        return {
            'nodes': nodes,
            'links': links
        }

# Initialize the generator
generator = HypothesisGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'ollama_model': generator.ollama_model,
        'services': {
            'ollama': 'available',
            'huggingface': 'available'
        }
    })

@app.route('/generate_hypotheses', methods=['POST'])
def generate_hypotheses():
    """
    Generate hypotheses from a premise.
    
    Expected JSON payload:
    {
        "premise": "Your premise statement",
        "context": "Additional context (optional)"
    }
    """
    try:
        data = request.get_json()
        if not data or 'premise' not in data:
            return jsonify({'error': 'Missing premise field'}), 400
        
        premise = data['premise']
        context = data.get('context', '')
        
        if not premise.strip():
            return jsonify({'error': 'Empty premise provided'}), 400
        
        # Generate hypotheses
        result = generator.generate_hypothesis(premise, context)
        
        if result['success']:
            # Generate knowledge graph
            knowledge_graph = generator.generate_knowledge_graph(
                premise, result['hypotheses']
            )
            result['knowledge_graph'] = knowledge_graph
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating hypotheses: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/test_nli', methods=['POST'])
def test_nli():
    """
    Test natural language inference between premise and hypothesis.
    
    Expected JSON payload:
    {
        "premise": "The premise statement",
        "hypothesis": "The hypothesis to test"
    }
    """
    try:
        data = request.get_json()
        if not data or 'premise' not in data or 'hypothesis' not in data:
            return jsonify({'error': 'Missing premise or hypothesis field'}), 400
        
        premise = data['premise']
        hypothesis = data['hypothesis']
        
        if not premise.strip() or not hypothesis.strip():
            return jsonify({'error': 'Empty premise or hypothesis provided'}), 400
        
        # Test NLI relationship
        result = generator.test_nli_relationship(premise, hypothesis)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error testing NLI: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/explore_knowledge', methods=['POST'])
def explore_knowledge():
    """
    Comprehensive knowledge exploration: generate hypotheses and test relationships.
    
    Expected JSON payload:
    {
        "premise": "Your premise statement",
        "context": "Additional context (optional)",
        "test_nli": true
    }
    """
    try:
        data = request.get_json()
        if not data or 'premise' not in data:
            return jsonify({'error': 'Missing premise field'}), 400
        
        premise = data['premise']
        context = data.get('context', '')
        test_nli = data.get('test_nli', False)
        
        if not premise.strip():
            return jsonify({'error': 'Empty premise provided'}), 400
        
        # Generate hypotheses
        hyp_result = generator.generate_hypothesis(premise, context)
        
        if not hyp_result['success']:
            return jsonify(hyp_result), 500
        
        # Test NLI relationships if requested
        if test_nli and hyp_result['hypotheses']:
            for hypothesis in hyp_result['hypotheses']:
                nli_result = generator.test_nli_relationship(premise, hypothesis['statement'])
                if nli_result['success']:
                    hypothesis['nli_label'] = nli_result['predicted_label']
                    hypothesis['nli_confidence'] = nli_result['confidence']
                    hypothesis['nli_scores'] = nli_result['all_scores']
        
        # Generate enhanced knowledge graph
        knowledge_graph = generator.generate_knowledge_graph(
            premise, hyp_result['hypotheses']
        )
        
        return jsonify({
            'success': True,
            'premise': premise,
            'context': context,
            'hypotheses': hyp_result['hypotheses'],
            'knowledge_graph': knowledge_graph,
            'nli_tested': test_nli
        })
        
    except Exception as e:
        logger.error(f"Error exploring knowledge: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Hypothesis Generation and Knowledge Exploration Server...")
    logger.info("Make sure Ollama is running: ollama serve")
    logger.info(f"Make sure model is available: ollama pull {generator.ollama_model}")
    app.run(host='0.0.0.0', port=5002, debug=True)