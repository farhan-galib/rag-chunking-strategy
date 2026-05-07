# RAG Chunking Strategy Project

A comprehensive implementation of various text chunking strategies for Retrieval-Augmented Generation (RAG) based AI solutions.

## 📚 Overview

This project demonstrates multiple chunking techniques used in RAG systems:

1. **Fixed-Size Chunking** - Simple, deterministic chunks of fixed length
2. **Sliding Window Chunking** - Overlapping chunks with configurable step size
3. **Semantic Chunking** - Context-aware chunking based on sentences and paragraphs
4. **Recursive/Hierarchical Chunking** - Structure-preserving chunking based on document sections
5. **Hybrid Chunking** - Combines multiple strategies for optimal results

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
python main.py
```

### Basic Usage

```python
from chunking_strategies import SemanticChunking
from datasets import DATASETS

# Get sample text
text = DATASETS['technical_faq']

# Create chunker
chunker = SemanticChunking(max_chunk_size=500)

# Generate chunks
chunks = chunker.chunk(text, source='technical_faq')

# Analyze chunks
from chunking_strategies import ChunkAnalyzer
analysis = ChunkAnalyzer.analyze_chunks(chunks)
print(f"Created {analysis['total_chunks']} chunks")
```

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