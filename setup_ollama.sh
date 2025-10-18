#!/bin/bash

# Setup script for Hypothesis Generator
echo "🔬 Setting up Hypothesis Generator with Ollama and RoBERTa"
echo "========================================================"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed. Please install it first:"
    echo "   Visit: https://ollama.ai"
    echo "   Or run: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
else
    echo "✅ Ollama is installed"
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "🚀 Starting Ollama server..."
    ollama serve &
    sleep 5
else
    echo "✅ Ollama server is running"
fi

# Pull the Qwen model
echo "📥 Pulling Qwen 0.5B model (this may take a few minutes)..."
ollama pull qwen2:0.5b

if [ $? -eq 0 ]; then
    echo "✅ Qwen 0.5B model downloaded successfully"
else
    echo "❌ Failed to download Qwen model"
    echo "💡 You can try manually: ollama pull qwen2:0.5b"
fi

# Check available models
echo ""
echo "📋 Available Ollama models:"
ollama list

echo ""
echo "🎉 Setup complete! You can now:"
echo "   1. Run the server: python hypothesis_server.py"
echo "   2. Open hypothesis_generator.html in your browser"
echo ""
echo "💡 Tips:"
echo "   - Make sure Ollama is running: ollama serve"
echo "   - Try different models: ollama pull llama3.2:1b"
echo "   - The app uses HuggingFace API for RoBERTa MNLI (no API key needed)"