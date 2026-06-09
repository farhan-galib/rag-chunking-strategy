# RAG Chunking Strategy Project

A comprehensive implementation of various text chunking strategies for Retrieval-Augmented Generation (RAG) based AI solutions with **intelligent dynamic recommendations**.

## 📚 Overview

This project demonstrates multiple chunking techniques used in RAG systems:

1. **Fixed-Size Chunking** - Simple, deterministic chunks of fixed length
2. **Sliding Window Chunking** - Overlapping chunks with configurable step size
3. **Semantic Chunking** - Context-aware chunking based on sentences and paragraphs
4. **Recursive/Hierarchical Chunking** - Structure-preserving chunking based on document sections
5. **Hybrid Chunking** - Combines multiple strategies for optimal results

## ✨ New Dynamic Features

- **Intelligent Analysis**: Automatically analyzes document characteristics (length, complexity, structure, semantic density)
- **Adaptive Recommendations**: Goes beyond simple document type detection to consider content-specific factors
- **Confidence Scoring**: Provides confidence levels for recommendations with alternative strategies
- **Parameter Optimization**: Dynamically adjusts chunking parameters based on content analysis
- **Content Insights**: Shows detailed analysis of what influenced the recommendation

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Command Line Demo

```bash
python main.py
```

### Web UI

Launch the interactive web interface for document analysis and chunking recommendations:

```bash
streamlit run ui.py
```

Then open http://localhost:8501 in your browser.

**Features:**
- Upload multiple text files (.txt, .md) for analysis
- Choose from pre-loaded sample datasets
- **Dynamic chunking strategy recommendations based on content analysis**
- **Confidence scores and alternative strategy suggestions**
- **Content analysis insights showing semantic density, structural complexity, etc.**
- **Business dashboard page with plain-English readiness summaries and chart-based insights**
- Detailed quality metrics and chunk previews

**Try it out:**
1. Use the included `sample_legal.txt` file to test document upload
2. Select different built-in datasets from the sidebar
3. Compare dynamic recommendations for different document types
4. Explore content analysis insights to understand recommendation reasoning

## 🎯 Web UI Features

- **Document Upload**: Upload multiple text files (.txt, .md) for analysis
- **Built-in Datasets**: Choose from pre-loaded sample datasets
- **Automatic Analysis**: Detect document type and content characteristics
- **Smart Recommendations**: Get personalized chunking strategy recommendations
- **Detailed Metrics**: View chunking quality scores and statistics
- **Sample Chunks**: Preview how your documents will be chunked

## 📊 Included Datasets

1. **Technical FAQ** - FAQ-style technical documentation
2. **Legal Document** - Structured legal/terms of service
3. **Medical Research** - Academic research paper with sections
4. **Code Documentation** - API documentation

## 📊 Included Datasets

1. **Technical FAQ** - FAQ-style technical documentation
2. **Legal Document** - Structured legal/terms of service
3. **Medical Research** - Academic research paper with sections
4. **Code Documentation** - API documentation
5. **E-commerce Products** - Product descriptions
6. **Support Conversations** - Customer support tickets

## 📈 Evaluation Metrics

Each chunking strategy is evaluated on:

- **Coherence Score** - Uniformity of chunk sizes
- **Semantic Preservation** - How well meaning is preserved
- **Retrieval Efficiency** - Optimal chunk size for retrieval
- **Boundary Quality** - Quality of chunk boundaries
- **Overall Quality** - Weighted combination of above metrics

## 🔧 Configuration

Each strategy accepts customizable parameters:

```python
# Fixed-size with 30% overlap
fixed = FixedSizeChunking(chunk_size=500, overlap=150)

# Semantic with size constraints
semantic = SemanticChunking(max_chunk_size=600, min_chunk_size=100)

# Sliding window with custom step
window = SlidingWindowChunking(window_size=400, step_size=200)

# Recursive with overlap
recursive = RecursiveChunking(max_chunk_size=1000, chunk_overlap=100)
```

## 📝 Best Practices

1. **Choose chunk size based on use case**
   - FAQ/Q&A: 300-400 chars
   - Long documents: 500-800 chars
   - Code: 200-400 chars

2. **Consider semantic boundaries**
   - Use semantic chunking for complex documents
   - Preserve section headers and structure

3. **Test with your data**
   - Use the evaluation metrics to compare strategies
   - Different data types may require different approaches

4. **Optimize for retrieval**
   - Balance chunk size with retrieval latency
   - Consider embedding model input limits

## 🎯 Use Cases

- **Document Retrieval Systems** - Split documents for semantic search
- **FAQ Chatbots** - Question-answer pair chunking
- **Code Documentation** - Structure-preserving chunking
- **Legal Analysis** - Preserve document hierarchy
- **Research Paper Analysis** - Section-aware chunking

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please feel free to submit PRs or open issues.

## 📧 Contact

For questions or suggestions, please reach out.