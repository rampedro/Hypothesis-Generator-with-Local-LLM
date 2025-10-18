# RoBERTa + RDF Attention Visualization

An enhanced D3.js visualization tool that combines RDF knowledge graphs with RoBERTa transformer attention weights to provide multi-layered semantic analysis and visual exploration of text.

## Features

- **RoBERTa Attention Analysis**: Extract and visualize attention weights from the RoBERTa-large-mnli model
- **RDF Integration**: Combine traditional RDF triples with attention-based relationships
- **Multi-layer Visualization**: Interactive exploration of different attention layers
- **LLM Enhancement**: Use GPT models to generate additional text, extract entities, and interpret attention patterns
- **Real-time Analysis**: Dynamic text input with immediate visualization updates

## Architecture

### Backend Components

1. **RoBERTa Attention Server** (`roberta_attention_server.py`)
   - Flask API server with RoBERTa model integration
   - Extracts attention weights, token embeddings, and semantic relationships
   - Combines RDF triples with attention-based connections

2. **LLM Integration** (`llm_integration.py`)
   - Supports GitHub Models and OpenAI APIs
   - Text expansion, entity extraction, and RDF generation
   - Semantic layer analysis and attention interpretation

### Frontend Components

3. **Enhanced Visualization** (`attention_visualization.html`)
   - Interactive D3.js visualization with multi-layer support
   - Real-time controls for attention thresholds and layer selection
   - Combined view of RDF and attention relationships

## Setup Instructions

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd /path/to/d3rdf

# The Python environment should already be configured
# If not, create a virtual environment:
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

The required packages should already be installed. If not:

```bash
pip install flask flask-cors torch transformers numpy scipy requests openai
```

### 3. LLM API Configuration (Optional)

For enhanced LLM features, set one of these environment variables:

**Option A: GitHub Models (Recommended for development)**
```bash
export GITHUB_TOKEN="your_github_personal_access_token"
```

**Option B: OpenAI API**
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

To get a GitHub Personal Access Token:
1. Go to GitHub Settings > Developer settings > Personal access tokens
2. Create a new token with appropriate permissions
3. Use the token as your GITHUB_TOKEN

### 4. Start the Backend Server

```bash
python roberta_attention_server.py
```

The server will start on `http://localhost:5000` and begin loading the RoBERTa model (this may take a few minutes on first run).

### 5. Open the Visualization

Open `attention_visualization.html` in a web browser, or serve it via a local HTTP server:

```bash
# Option 1: Direct file opening
open attention_visualization.html

# Option 2: Local HTTP server (recommended)
python -m http.server 8000
# Then open http://localhost:8000/attention_visualization.html
```

## Usage Guide

### Basic Usage

1. **Start with RDF**: Click "Load RDF Only" to see the default RDF graph
2. **Add Text Analysis**: Enter text in the input field and click "Analyze with RoBERTa"
3. **Combine Views**: Use "Combine RDF + Attention" to see both relationship types
4. **Explore Layers**: Use the layer selector to view attention from different transformer layers

### Advanced Features

#### Attention Threshold
- Adjust the slider to filter attention relationships by weight
- Higher thresholds show only the strongest attention connections
- Lower thresholds reveal more subtle relationships

#### Visualization Modes
- **Combined View**: Shows both RDF and attention relationships
- **RDF Only**: Traditional RDF knowledge graph
- **Attention Only**: Pure attention-based relationships from text

#### Layer Selection
- View attention patterns from different transformer layers
- Middle layers often capture semantic relationships
- Early layers focus on syntax, later layers on semantics

### API Endpoints

The backend provides several REST API endpoints:

- `GET /health` - Server health check
- `POST /analyze` - Full RoBERTa attention analysis
- `POST /relationships` - Extract attention-based relationships
- `POST /rdf_plus_attention` - Combine RDF with attention data
- `POST /llm_generate_text` - Expand text using LLM (requires API key)
- `POST /llm_extract_entities` - Extract entities using LLM
- `POST /llm_generate_rdf` - Generate RDF from text using LLM
- `POST /enhanced_analysis` - Comprehensive analysis with all features

## Example Analysis

Try analyzing this sample text:

```
Thailand has delicious food. Tom Yum Kung is a spicy soup that includes shrimp, chili, and lemon. The lemon gives it a sour taste while chili makes it spicy. This dish represents the complex flavors of Thai cuisine.
```

The visualization will show:
- **RDF relationships**: Traditional knowledge graph connections
- **Attention relationships**: How words attend to each other in the transformer
- **Multi-layer insights**: Different semantic patterns at various depths
- **LLM enhancement**: Additional context and entity extraction

## Troubleshooting

### Common Issues

1. **Model Loading Slow**: The RoBERTa model is large (~1.4GB). First download may take time.

2. **CORS Errors**: Make sure the Flask server is running and CORS is enabled.

3. **LLM Features Disabled**: Set the appropriate environment variables for API access.

4. **Memory Issues**: The RoBERTa model requires significant RAM. Close other applications if needed.

### Debug Mode

Run the server in debug mode for detailed logging:

```bash
FLASK_DEBUG=1 python roberta_attention_server.py
```

## Technical Details

### Model Information
- **RoBERTa Model**: FacebookAI/roberta-large-mnli
- **Input Limit**: 512 tokens per analysis
- **Attention Heads**: 16 heads per layer, 24 layers total
- **Output**: Attention matrices, token embeddings, semantic relationships

### Visualization Features
- **Force-directed layout**: Automatic node positioning
- **Interactive controls**: Real-time parameter adjustment
- **Multi-layer support**: View different transformer layers
- **Responsive design**: Adapts to different screen sizes

### Performance Considerations
- **GPU Support**: Automatically uses CUDA if available
- **Batch Processing**: Single text analysis per request
- **Caching**: Model loaded once and reused for multiple requests

## Future Enhancements

Potential areas for expansion:
- Support for longer texts (chunking and aggregation)
- Additional transformer models (BERT, GPT, etc.)
- Interactive attention matrix visualization
- Export capabilities for analysis results
- Batch processing for multiple texts
- Integration with knowledge bases

## License

Based on the original D3.js RDF visualization by Rathachai Chawuthai, enhanced with modern transformer attention analysis capabilities.

## Description
This project uses force layout of D3<sup>1</sup> to visualize the expression "subject–predicate–object", a.k.a. a triple in RDF terminology, under semantic web and linked data domains. It is implemented on top of the Force-Directed Graph<sup>2</sup> by adding a label for every node (resource: subject or object), and an arrow and a label for every edge (predicate).

## Codes
- Visualization of Triples : http://rathachai.github.io/d3rdf/index.html
- Curve Links : http://rathachai.github.io/d3rdf/curvelinks.html
- User Adding Triples : http://rathachai.github.io/d3rdf/userAdding.html

## **Announcement**
- The new version for Colab https://github.com/Rathachai/rdfviz/blob/main/ex/rdfviz-examples.ipynb

## Example
![alt tag](https://raw.github.com/rathachai/d3rdf/master/images/simpletriples.png)

## Citation
**Please refer to :**

Chawuthai, R., & Takeda, H.: Rdf graph visualization by interpreting linked data as knowledge. In Joint International Semantic Technology Conference (pp. 23-39). Springer, 2015.

## References
0. D3 : http://d3js.org/
0. Force-Directed Graph : http://bl.ocks.org/mbostock/4062045
0. Curve Links : http://bl.ocks.org/mbostock/4600693
