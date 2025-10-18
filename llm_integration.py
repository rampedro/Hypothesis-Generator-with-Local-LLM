#!/usr/bin/env python3
"""
LLM Integration Module for Text Generation and Semantic Analysis
Integrates with OpenAI and GitHub Models for enhanced text processing
"""

import openai
import json
from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)

class LLMTextAnalyzer:
    def __init__(self, api_key: str = None, base_url: str = "https://models.inference.ai.azure.com"):
        """
        Initialize LLM analyzer with GitHub Models or OpenAI API.
        
        Args:
            api_key: GitHub PAT or OpenAI API key
            base_url: API base URL (GitHub Models by default)
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "gpt-4o-mini"  # Default GitHub model
        
    def generate_text_expansion(self, text: str, context: str = "") -> str:
        """
        Generate expanded text with additional context and details.
        
        Args:
            text: Original text to expand
            context: Additional context for expansion
            
        Returns:
            Expanded text with more details
        """
        prompt = f"""
        Please expand the following text with more details, context, and related information. 
        Keep the core meaning but add relevant background, explanations, and connections.
        
        Original text: {text}
        
        Additional context: {context}
        
        Expanded text:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating text expansion: {e}")
            return text
    
    def extract_entities_and_relationships(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships from text using LLM.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing entities and relationships
        """
        prompt = f"""
        Analyze the following text and extract:
        1. Key entities (people, places, things, concepts)
        2. Relationships between entities
        3. Semantic connections
        
        Format your response as JSON with this structure:
        {{
            "entities": [
                {{"name": "entity_name", "type": "person|place|thing|concept", "description": "brief description"}}
            ],
            "relationships": [
                {{"source": "entity1", "target": "entity2", "relationship": "relationship_type", "description": "explanation"}}
            ]
        }}
        
        Text to analyze: {text}
        
        JSON Response:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"entities": [], "relationships": []}
                
        except Exception as e:
            logger.error(f"Error extracting entities and relationships: {e}")
            return {"entities": [], "relationships": []}
    
    def generate_rdf_triples(self, text: str) -> List[Dict[str, str]]:
        """
        Generate RDF triples from text using LLM.
        
        Args:
            text: Text to convert to RDF
            
        Returns:
            List of RDF triple dictionaries
        """
        prompt = f"""
        Convert the following text into RDF triples. Use appropriate prefixes like ex: for custom entities.
        
        Format each triple as: {{"subject": "ex:Subject", "predicate": "ex:predicate", "object": "ex:Object"}}
        
        Focus on:
        - Main entities and their types
        - Properties and attributes
        - Relationships between entities
        - Hierarchical connections
        
        Return only a JSON array of triples, no other text.
        
        Text: {text}
        
        RDF Triples:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON array from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error generating RDF triples: {e}")
            return []
    
    def generate_semantic_layers(self, text: str) -> Dict[str, Any]:
        """
        Generate different semantic layers of analysis for the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing different layers of semantic analysis
        """
        prompt = f"""
        Analyze the text from multiple semantic layers and provide insights for each layer:
        
        1. Syntactic Layer: Grammar structure, dependencies
        2. Semantic Layer: Meaning, concepts, themes
        3. Pragmatic Layer: Context, implications, intentions
        4. Discourse Layer: Text structure, coherence, flow
        5. Conceptual Layer: Abstract concepts, knowledge domains
        
        Format as JSON:
        {{
            "syntactic": {{"description": "...", "key_elements": ["..."]}},
            "semantic": {{"description": "...", "key_concepts": ["..."]}},
            "pragmatic": {{"description": "...", "implications": ["..."]}},
            "discourse": {{"description": "...", "structure": ["..."]}},
            "conceptual": {{"description": "...", "domains": ["..."]}}
        }}
        
        Text: {text}
        
        Analysis:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.4
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error generating semantic layers: {e}")
            return {}
    
    def enhance_attention_interpretation(self, attention_data: List[Dict], text: str) -> Dict[str, Any]:
        """
        Use LLM to interpret attention patterns and provide semantic meaning.
        
        Args:
            attention_data: Attention relationships from RoBERTa
            text: Original text
            
        Returns:
            Enhanced interpretation of attention patterns
        """
        # Format attention data for LLM
        attention_summary = []
        for rel in attention_data[:10]:  # Limit to top 10 for context
            attention_summary.append(f"'{rel['source']}' attends to '{rel['target']}' (weight: {rel['weight']:.3f})")
        
        prompt = f"""
        Analyze the following attention patterns from a RoBERTa transformer model and provide semantic interpretation:
        
        Original text: {text}
        
        Attention patterns:
        {chr(10).join(attention_summary)}
        
        Please provide:
        1. Semantic meaning of high-attention connections
        2. Linguistic phenomena captured (dependencies, coreference, etc.)
        3. Conceptual relationships identified
        4. Potential implications for understanding
        
        Format as JSON:
        {{
            "semantic_interpretation": "...",
            "linguistic_phenomena": ["..."],
            "conceptual_relationships": ["..."],
            "implications": ["..."]
        }}
        
        Analysis:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.4
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error enhancing attention interpretation: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Example with GitHub Models (requires GitHub PAT)
    # analyzer = LLMTextAnalyzer(api_key="your_github_pat")
    
    # Example with OpenAI (requires OpenAI API key)
    # analyzer = LLMTextAnalyzer(
    #     api_key="your_openai_key", 
    #     base_url="https://api.openai.com/v1"
    # )
    
    sample_text = "Thailand has delicious food. Tom Yum Kung is a spicy soup."
    
    # Test without API key (will fail gracefully)
    analyzer = LLMTextAnalyzer()
    
    print("Testing LLM Text Analyzer...")
    print("Note: This will fail without proper API key, but shows the interface")
    
    # entities = analyzer.extract_entities_and_relationships(sample_text)
    # print(f"Entities: {entities}")
    
    # rdf_triples = analyzer.generate_rdf_triples(sample_text)
    # print(f"RDF Triples: {rdf_triples}")