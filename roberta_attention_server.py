#!/usr/bin/env python3
"""
RoBERTa Attention Visualization Server
Integrates RoBERTa-large-mnli model for attention weight extraction
and semantic analysis with D3.js visualization.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Any
import logging
import os
from llm_integration import LLMTextAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

class RoBERTaAttentionAnalyzer:
    def __init__(self):
        """Initialize RoBERTa model and tokenizer."""
        self.model_name = "FacebookAI/roberta-large-mnli"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaModel.from_pretrained(
            self.model_name, 
            output_attentions=True,
            output_hidden_states=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("RoBERTa model loaded successfully")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text and extract attention weights, token information, and embeddings.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing tokens, attention weights, embeddings, and metadata
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract components
        attention_weights = outputs.attentions  # Tuple of attention matrices for each layer
        hidden_states = outputs.hidden_states   # Hidden states for each layer
        last_hidden_state = outputs.last_hidden_state
        
        # Convert tokens back to words
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Process attention weights (average across heads for visualization)
        attention_data = []
        for layer_idx, layer_attention in enumerate(attention_weights):
            # Shape: [batch_size, num_heads, seq_len, seq_len]
            # Average across attention heads
            avg_attention = layer_attention[0].mean(dim=0).cpu().numpy()
            attention_data.append({
                'layer': layer_idx,
                'attention_matrix': avg_attention.tolist(),
                'shape': list(avg_attention.shape)
            })
        
        # Extract token embeddings from last layer
        embeddings = last_hidden_state[0].cpu().numpy()
        
        # Create token information with positions
        token_info = []
        for i, token in enumerate(tokens):
            token_info.append({
                'index': i,
                'token': token,
                'embedding': embeddings[i].tolist(),
                'is_special': token in ['<s>', '</s>', '<pad>']
            })
        
        return {
            'tokens': token_info,
            'attention_layers': attention_data,
            'num_layers': len(attention_weights),
            'num_heads': attention_weights[0].shape[1],
            'sequence_length': len(tokens),
            'original_text': text
        }
    
    def extract_semantic_relationships(self, text: str, threshold: float = 0.1) -> List[Dict]:
        """
        Extract semantic relationships based on attention weights.
        
        Args:
            text: Input text
            threshold: Minimum attention weight to consider as a relationship
            
        Returns:
            List of relationship dictionaries
        """
        analysis = self.analyze_text(text)
        relationships = []
        
        # Use attention from middle layers (they often capture semantic relationships)
        middle_layer_idx = len(analysis['attention_layers']) // 2
        attention_matrix = np.array(analysis['attention_layers'][middle_layer_idx]['attention_matrix'])
        
        tokens = [token['token'] for token in analysis['tokens']]
        
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if i != j and attention_matrix[i][j] > threshold:
                    # Skip special tokens
                    if not (analysis['tokens'][i]['is_special'] or analysis['tokens'][j]['is_special']):
                        relationships.append({
                            'source': tokens[i],
                            'target': tokens[j],
                            'weight': float(attention_matrix[i][j]),
                            'layer': middle_layer_idx,
                            'source_idx': i,
                            'target_idx': j
                        })
        
        return relationships

# Initialize the analyzer and LLM integration
analyzer = RoBERTaAttentionAnalyzer()

# Initialize LLM analyzer (will work if API key is provided via environment)
llm_analyzer = None
try:
    github_token = os.getenv('GITHUB_TOKEN')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if github_token:
        llm_analyzer = LLMTextAnalyzer(api_key=github_token)
        logger.info("LLM analyzer initialized with GitHub Models")
    elif openai_key:
        llm_analyzer = LLMTextAnalyzer(
            api_key=openai_key, 
            base_url="https://api.openai.com/v1"
        )
        logger.info("LLM analyzer initialized with OpenAI")
    else:
        logger.warning("No API key found for LLM integration. LLM features will be disabled.")
except Exception as e:
    logger.warning(f"Failed to initialize LLM analyzer: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model': analyzer.model_name})

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Analyze text and return attention weights and token information.
    
    Expected JSON payload:
    {
        "text": "Your text to analyze"
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Analyze text
        result = analyzer.analyze_text(text)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/relationships', methods=['POST'])
def extract_relationships():
    """
    Extract semantic relationships from text based on attention weights.
    
    Expected JSON payload:
    {
        "text": "Your text to analyze",
        "threshold": 0.1  (optional, default 0.1)
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        threshold = data.get('threshold', 0.1)
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Extract relationships
        relationships = analyzer.extract_semantic_relationships(text, threshold)
        
        return jsonify({
            'success': True,
            'relationships': relationships,
            'text': text,
            'threshold': threshold
        })
        
    except Exception as e:
        logger.error(f"Error extracting relationships: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/rdf_plus_attention', methods=['POST'])
def combine_rdf_attention():
    """
    Combine RDF triples with attention-based relationships for enhanced visualization.
    
    Expected JSON payload:
    {
        "text": "Your text to analyze",
        "rdf_triples": [
            {"subject": "ex:A", "predicate": "ex:relates", "object": "ex:B"}
        ],
        "attention_threshold": 0.1
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '')
        rdf_triples = data.get('rdf_triples', [])
        threshold = data.get('attention_threshold', 0.1)
        
        result = {
            'rdf_triples': rdf_triples,
            'attention_relationships': [],
            'combined_graph': {'nodes': [], 'links': []}
        }
        
        # If text is provided, analyze attention
        if text.strip():
            relationships = analyzer.extract_semantic_relationships(text, threshold)
            result['attention_relationships'] = relationships
            
            # Create combined graph structure for D3.js
            all_entities = set()
            
            # Add RDF entities
            for triple in rdf_triples:
                all_entities.add(triple['subject'])
                all_entities.add(triple['object'])
            
            # Add attention entities
            for rel in relationships:
                all_entities.add(rel['source'])
                all_entities.add(rel['target'])
            
            # Create nodes
            nodes = []
            for entity in all_entities:
                nodes.append({
                    'id': entity,
                    'label': entity,
                    'type': 'rdf' if entity.startswith('ex:') else 'token',
                    'weight': 1
                })
            
            # Create links
            links = []
            
            # Add RDF links
            for triple in rdf_triples:
                links.append({
                    'source': triple['subject'],
                    'target': triple['object'],
                    'predicate': triple['predicate'],
                    'type': 'rdf',
                    'weight': 1
                })
            
            # Add attention links
            for rel in relationships:
                links.append({
                    'source': rel['source'],
                    'target': rel['target'],
                    'predicate': 'attention',
                    'type': 'attention',
                    'weight': rel['weight'],
                    'layer': rel['layer']
                })
            
            result['combined_graph'] = {
                'nodes': nodes,
                'links': links
            }
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error combining RDF and attention: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/llm_generate_text', methods=['POST'])
def llm_generate_text():
    """
    Generate expanded text using LLM.
    
    Expected JSON payload:
    {
        "text": "Your text to expand",
        "context": "Additional context (optional)"
    }
    """
    if not llm_analyzer:
        return jsonify({'error': 'LLM integration not available. Please set GITHUB_TOKEN or OPENAI_API_KEY environment variable.'}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        context = data.get('context', '')
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Generate expanded text
        expanded_text = llm_analyzer.generate_text_expansion(text, context)
        
        return jsonify({
            'success': True,
            'original_text': text,
            'expanded_text': expanded_text,
            'context': context
        })
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/llm_extract_entities', methods=['POST'])
def llm_extract_entities():
    """
    Extract entities and relationships using LLM.
    
    Expected JSON payload:
    {
        "text": "Your text to analyze"
    }
    """
    if not llm_analyzer:
        return jsonify({'error': 'LLM integration not available. Please set GITHUB_TOKEN or OPENAI_API_KEY environment variable.'}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Extract entities and relationships
        entities_and_relationships = llm_analyzer.extract_entities_and_relationships(text)
        
        return jsonify({
            'success': True,
            'text': text,
            'entities_and_relationships': entities_and_relationships
        })
        
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/llm_generate_rdf', methods=['POST'])
def llm_generate_rdf():
    """
    Generate RDF triples from text using LLM.
    
    Expected JSON payload:
    {
        "text": "Your text to convert to RDF"
    }
    """
    if not llm_analyzer:
        return jsonify({'error': 'LLM integration not available. Please set GITHUB_TOKEN or OPENAI_API_KEY environment variable.'}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Generate RDF triples
        rdf_triples = llm_analyzer.generate_rdf_triples(text)
        
        return jsonify({
            'success': True,
            'text': text,
            'rdf_triples': rdf_triples
        })
        
    except Exception as e:
        logger.error(f"Error generating RDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/llm_semantic_layers', methods=['POST'])
def llm_semantic_layers():
    """
    Generate semantic layer analysis using LLM.
    
    Expected JSON payload:
    {
        "text": "Your text to analyze"
    }
    """
    if not llm_analyzer:
        return jsonify({'error': 'LLM integration not available. Please set GITHUB_TOKEN or OPENAI_API_KEY environment variable.'}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Generate semantic layers
        semantic_layers = llm_analyzer.generate_semantic_layers(text)
        
        return jsonify({
            'success': True,
            'text': text,
            'semantic_layers': semantic_layers
        })
        
    except Exception as e:
        logger.error(f"Error generating semantic layers: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/enhanced_analysis', methods=['POST'])
def enhanced_analysis():
    """
    Perform comprehensive analysis combining RoBERTa attention and LLM insights.
    
    Expected JSON payload:
    {
        "text": "Your text to analyze",
        "attention_threshold": 0.1,
        "generate_rdf": true,
        "expand_text": false
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        threshold = data.get('attention_threshold', 0.1)
        generate_rdf = data.get('generate_rdf', False)
        expand_text = data.get('expand_text', False)
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        result = {'original_text': text}
        
        # RoBERTa attention analysis
        attention_analysis = analyzer.analyze_text(text)
        attention_relationships = analyzer.extract_semantic_relationships(text, threshold)
        
        result['attention_analysis'] = attention_analysis
        result['attention_relationships'] = attention_relationships
        
        # LLM analysis (if available)
        if llm_analyzer:
            try:
                if expand_text:
                    result['expanded_text'] = llm_analyzer.generate_text_expansion(text)
                
                if generate_rdf:
                    result['llm_rdf_triples'] = llm_analyzer.generate_rdf_triples(text)
                
                result['entities_and_relationships'] = llm_analyzer.extract_entities_and_relationships(text)
                result['semantic_layers'] = llm_analyzer.generate_semantic_layers(text)
                result['attention_interpretation'] = llm_analyzer.enhance_attention_interpretation(
                    attention_relationships, text
                )
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
                result['llm_error'] = str(e)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting RoBERTa Attention Visualization Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)