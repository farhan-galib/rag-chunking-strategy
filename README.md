# RAG Chunking Strategy Project

A working demo of document chunking strategies for Retrieval-Augmented Generation (RAG) systems with dynamic strategy recommendations.

## 📚 Overview

This repository implements several chunking techniques used in RAG applications:

1. **Fixed-Size Chunking** - deterministic chunks of fixed length
2. **Sliding Window Chunking** - overlapping chunks with configurable step size
3. **Semantic Chunking** - content-aware chunking based on sentence and paragraph structure
4. **Recursive/Hierarchical Chunking** - structure-preserving chunking that respects document sections
5. **Hybrid Chunking** - combines multiple strategies for balanced results

## ✨ What’s Included

- **Interactive Streamlit UI** in `ui.py`
- **Command-line demo** in `main.py`
- **Multi-format upload support** for TXT, MD, PDF, DOCX, XLSX, and XLS
- **Dynamic recommendations** with confidence scoring and alternative strategies
- **Content analysis insights** including semantic preservation, boundary quality, and retrieval readiness
- **Built-in text datasets** for quick experimentation

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Command Line Demo

Run the terminal demo to exercise the chunking strategies across sample datasets:

```bash
python main.py
```

### Web UI

Start the interactive Streamlit app:

```bash
streamlit run ui.py
```

Then open `http://localhost:8501` in your browser.

## 🎛️ Supported Inputs

The app supports uploading:

- `.txt`
- `.md`
- `.pdf`
- `.docx`
- `.xlsx`
- `.xls`

It also lets you choose from built-in sample datasets loaded from `datasets/*.txt`.

## 📦 Built-in Sample Datasets

The following datasets are available in the app sidebar:

- `technical_faq`
- `legal_document`
- `medical_research`
- `code_documentation`
- `ecommerce_products`
- `support_conversations`

## 🧪 What the App Does

- Upload or select sample documents
- Automatically detect document characteristics
- Recommend the best chunking strategy for RAG workflows
- Show confidence and alternative strategy options
- Display chunk-level analysis and readiness summaries

## 📊 Evaluation Metrics

Each strategy is evaluated using:

- **Overall Quality** - aggregate chunking effectiveness
- **Semantic Preservation** - meaning retention across chunks
- **Boundary Quality** - chunk cut point quality
- **Retrieval Efficiency** - expected retrieval performance

## 🔧 Extending the Code

You can adjust strategy parameters in code:

```python
fixed = FixedSizeChunking(chunk_size=500, overlap=150)
semantic = SemanticChunking(max_chunk_size=600, min_chunk_size=100)
window = SlidingWindowChunking(window_size=400, step_size=200)
recursive = RecursiveChunking(max_chunk_size=1000, chunk_overlap=100)
```

## 💡 Notes

- `main.py` is a CLI demonstration that runs through the available text datasets.
- `ui.py` is the Streamlit application for uploading files and viewing interactive chunking recommendations.
- `datasets/` contains `.txt` sample datasets; additional PDF files are included for future extraction or testing workflows.

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please submit PRs or open issues.

## 📧 Contact

For questions or suggestions, feel free to reach out.