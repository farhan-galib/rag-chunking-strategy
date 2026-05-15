"""
RAG Chunking Strategy UI
Web interface for document analysis and chunking recommendations
"""

import streamlit as st
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

from chunking_strategies import (
    FixedSizeChunking, SlidingWindowChunking, SemanticChunking,
    RecursiveChunking, HybridChunking, ChunkAnalyzer
)
from datasets_v2 import load_dataset, list_datasets
from evaluation import ChunkEvaluator
from dynamic_recommender import DynamicChunkingRecommender

# Document analysis is now handled by DynamicChunkingRecommender

def analyze_text(text: str) -> Dict:
    """Comprehensive text analysis with dynamic recommendations"""
    # Initialize dynamic recommender
    recommender = DynamicChunkingRecommender()

    # Get dynamic recommendation
    recommendation_result = recommender.recommend_strategy(text)

    # Apply recommended chunking
    chunks = recommendation_result['strategy'].chunk(text, source='uploaded')

    # Evaluate
    metrics = ChunkEvaluator.evaluate_strategy(chunks, len(text))
    analysis = ChunkAnalyzer.analyze_chunks(chunks)

    return {
        'doc_type': recommendation_result['doc_type'],
        'recommendation': recommendation_result,
        'chunks': chunks,
        'metrics': metrics,
        'analysis': analysis,
        'text_length': len(text)
    }


def get_rag_readiness(overall_quality: float, confidence: float) -> str:
    """Return a simple readiness label for business users."""
    if overall_quality >= 0.75 and confidence >= 0.7:
        return "Good"
    if overall_quality >= 0.6 and confidence >= 0.55:
        return "Fair"
    return "Needs Improvement"


def generate_business_summary(analysis: Dict) -> str:
    """Generate a plain-English summary for the dashboard."""
    readiness = get_rag_readiness(analysis['metrics'].overall_quality, analysis['recommendation']['confidence'])
    strategy = analysis['recommendation']['strategy_name'].replace('_', ' ').title()
    doc_type = analysis['doc_type'].title()
    overall = analysis['metrics'].overall_quality
    confidence = analysis['recommendation']['confidence']

    summary = [
        f"This document is classified as **{doc_type}** content.",
        f"We recommend using **{strategy}** for the best RAG integration.",
        f"The document is rated **{readiness}** for RAG integration with {confidence:.0%} confidence."
    ]

    if overall < 0.6:
        summary.append("The text may need better chunk boundaries or cleaner structure before it is optimal for retrieval-based use.")
    elif overall < 0.75:
        summary.append("The document is usable for RAG, but some tuning may improve retrieval accuracy.")
    else:
        summary.append("This content looks well suited for RAG with minimal additional work.")

    if analysis['analysis']['avg_chunk_size'] < 250:
        summary.append("The chunks are smaller than ideal, so retrieval may require more lookups.")
    if analysis['metrics'].boundary_quality < 0.6:
        summary.append("Chunk boundaries may not align well with natural text structure.")

    return " ".join(summary)


def render_business_dashboard(analyses: List[Dict]):
    """Render the business-facing insights dashboard."""
    st.header("📈 RAG Readiness Dashboard")
    st.markdown("This page summarizes document readiness for RAG integration in plain English.")

    if not analyses:
        st.info("Upload a document or select a built-in dataset to see dashboard insights.")
        return

    # Summary cards
    total_docs = len(analyses)
    avg_quality = sum(a['metrics'].overall_quality for a in analyses) / total_docs
    avg_confidence = sum(a['recommendation']['confidence'] for a in analyses) / total_docs
    readiness_labels = [get_rag_readiness(a['metrics'].overall_quality, a['recommendation']['confidence']) for a in analyses]
    readiness_counts = Counter(readiness_labels)
    strategy_counts = Counter(a['recommendation']['strategy_name'].replace('_', ' ').title() for a in analyses)
    type_counts = Counter(a['doc_type'].title() for a in analyses)

    with st.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents Reviewed", total_docs)
        c2.metric("Average RAG Quality", f"{avg_quality:.3f}")
        c3.metric("Average Confidence", f"{avg_confidence:.1%}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most common document types")
        st.vega_lite_chart({
            'data': {'values': [{'type': t, 'count': c} for t, c in type_counts.items()]},
            'mark': 'bar',
            'encoding': {
                'x': {'field': 'type', 'type': 'ordinal', 'title': 'Document Type'},
                'y': {'field': 'count', 'type': 'quantitative', 'title': 'Count'}
            }
        }, use_container_width=True)

    with col2:
        st.subheader("Strategy recommendations")
        st.vega_lite_chart({
            'data': {'values': [{'strategy': s, 'count': c} for s, c in strategy_counts.items()]},
            'mark': 'bar',
            'encoding': {
                'x': {'field': 'strategy', 'type': 'ordinal', 'title': 'Recommended Strategy'},
                'y': {'field': 'count', 'type': 'quantitative', 'title': 'Count'}
            }
        }, use_container_width=True)

    st.markdown("---")

    st.subheader("Plain English Business Insights")
    for analysis in analyses:
        filename = analysis.get('filename', 'Document')
        with st.expander(f"{filename}: {analysis['doc_type'].title()} | {get_rag_readiness(analysis['metrics'].overall_quality, analysis['recommendation']['confidence'])}"):
            st.write(generate_business_summary(analysis))

    st.markdown("---")
    st.subheader("Top concerns and next steps")

    concerns = []
    if avg_quality < 0.6:
        concerns.append("Some documents may require text cleanup, stronger paragraph structure, or improved chunk boundaries before RAG use.")
    if readiness_counts['Good'] == 0:
        concerns.append("No documents are currently in the top readiness tier; review chunk boundaries and document structure.")
    if type_counts.get('Legal', 0) > 0 and strategy_counts.get('Recursive', 0) == 0:
        concerns.append("Legal documents typically benefit from hierarchical chunking; verify the recommendation for those files.")

    if concerns:
        for concern in concerns:
            st.warning(concern)
    else:
        st.success("The uploaded documents are generally in good shape for RAG integration.")


def main():
    st.set_page_config(page_title="RAG Chunking Strategy Analyzer", page_icon="📄", layout="wide")

    st.sidebar.header("Page")
    page = st.sidebar.radio("Choose a view:", ["Upload & Analyze", "Business Dashboard"])

    st.title("📄 RAG Chunking Strategy Analyzer")
    st.markdown("Analyze documents and get intelligent chunking recommendations for RAG applications.")

    st.sidebar.header("📚 Built-in Datasets")
    available_datasets = list_datasets()
    selected_dataset = st.sidebar.selectbox(
        "Choose a built-in dataset:",
        ["None"] + available_datasets,
        key="built_in_dataset_selectbox"
    )

    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload text files for analysis",
        type=['txt', 'md'],
        accept_multiple_files=True
    )

    file_data = {}
    analyses = []

    if uploaded_files:
        st.subheader("📋 Uploaded Files")
        for uploaded_file in uploaded_files:
            text = uploaded_file.read().decode('utf-8')
            file_data[uploaded_file.name] = text

    if selected_dataset != "None":
        try:
            dataset_text = load_dataset(selected_dataset)
            file_data[selected_dataset] = dataset_text
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    for filename, text in file_data.items():
        analysis = analyze_text(text)
        analysis['filename'] = filename
        analyses.append(analysis)

    if page == "Upload & Analyze":
        if not file_data:
            st.info("Upload a text file or select a built-in dataset to begin analysis.")
        else:
            st.header("🔍 Analysis Results")
            for filename, text in file_data.items():
                analysis = next((a for a in analyses if a['filename'] == filename), None)
                if analysis is None:
                    continue

                with st.expander(f"📄 {filename}", expanded=True):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.subheader("Document Info")
                        st.metric("Text Length", f"{analysis['text_length']} chars")
                        st.write(f"**Detected Type:** {analysis['doc_type'].title()}")

                    with col2:
                        st.subheader("🎯 Dynamic Recommendation")
                        rec = analysis['recommendation']

                        confidence = rec.get('confidence', 0.5)
                        if confidence > 0.8:
                            st.success(f"**High Confidence:** {rec['strategy_name'].replace('_', ' ').title()}")
                        elif confidence > 0.6:
                            st.info(f"**Recommended:** {rec['strategy_name'].replace('_', ' ').title()}")
                        else:
                            st.warning(f"**Suggested:** {rec['strategy_name'].replace('_', ' ').title()}")

                        st.metric("Confidence", f"{confidence:.1%}")
                        st.write(f"**Reasoning:** {rec.get('reasoning', 'Based on content analysis')}")

                        if rec.get('alternatives'):
                            with st.expander("Alternative Strategies"):
                                for alt in rec['alternatives'][:3]:
                                    st.write(f"• **{alt['name'].replace('_', ' ').title()}** ({alt['score']:.1%})")
                                    st.caption(alt.get('reason', ''))

                        with st.expander("Content Analysis Insights"):
                            analysis_data = rec.get('analysis', {})
                            if analysis_data:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Semantic Density", f"{analysis_data.get('semantic_density', 0):.2f}")
                                    st.metric("Structural Complexity", f"{analysis_data.get('structural_complexity', 0):.2f}")
                                    st.metric("Avg Sentence Length", f"{analysis_data.get('avg_sentence_length', 0):.1f} chars")
                                with col_b:
                                    st.metric("Vocabulary Complexity", f"{analysis_data.get('vocabulary_complexity', 0):.2f}")
                                    st.metric("Has Headers", "Yes" if analysis_data.get('has_headers') else "No")
                                    st.metric("Code Blocks", analysis_data.get('code_blocks', 0))

                    st.subheader("📊 Chunking Metrics")
                    metrics = analysis['metrics']
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric("Overall Quality", f"{metrics.overall_quality:.3f}")
                    with m_col2:
                        st.metric("Coherence", f"{metrics.coherence_score:.3f}")
                    with m_col3:
                        st.metric("Boundary Quality", f"{metrics.boundary_quality:.3f}")

                    st.subheader("📈 Chunk Analysis")
                    chunk_analysis = analysis['analysis']
                    ca_col1, ca_col2, ca_col3, ca_col4 = st.columns(4)
                    with ca_col1:
                        st.metric("Total Chunks", chunk_analysis['total_chunks'])
                    with ca_col2:
                        st.metric("Avg Size", f"{chunk_analysis['avg_chunk_size']:.1f} chars")
                    with ca_col3:
                        st.metric("Min Size", f"{chunk_analysis['min_chunk_size']} chars")
                    with ca_col4:
                        st.metric("Max Size", f"{chunk_analysis['max_chunk_size']} chars")

                    st.subheader("📝 Sample Chunks")
                    sample_chunks = analysis['chunks'][:3]
                    for i, chunk in enumerate(sample_chunks):
                        with st.expander(f"Chunk {i+1} ({len(chunk.text)} chars)", expanded=False):
                            st.text(chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text)

    else:
        render_business_dashboard(analyses)

    #st.markdown("---")
    #st.markdown("Built with Streamlit for intelligent RAG chunking strategy analysis.")
    # Built-in dataset analysis
    if selected_dataset != "None":
        st.header(f"🔍 Built-in Dataset: {selected_dataset.upper()}")

        try:
            text = load_dataset(selected_dataset)
            analysis = analyze_text(text)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Document Info")
                st.metric("Text Length", f"{analysis['text_length']} chars")
                st.write(f"**Detected Type:** {analysis['doc_type'].title()}")

            with col2:
                st.subheader("🎯 Dynamic Recommendation")
                rec = analysis['recommendation']

                # Display confidence level
                confidence = rec.get('confidence', 0.5)
                if confidence > 0.8:
                    st.success(f"**High Confidence:** {rec['strategy_name'].replace('_', ' ').title()}")
                elif confidence > 0.6:
                    st.info(f"**Recommended:** {rec['strategy_name'].replace('_', ' ').title()}")
                else:
                    st.warning(f"**Suggested:** {rec['strategy_name'].replace('_', ' ').title()}")

                st.metric("Confidence", f"{confidence:.1%}")
                st.write(f"**Reasoning:** {rec.get('reasoning', 'Based on content analysis')}")

                # Show alternative strategies
                if rec.get('alternatives'):
                    with st.expander("Alternative Strategies"):
                        for alt in rec['alternatives'][:3]:  # Show top 3 alternatives
                            st.write(f"• **{alt['name'].replace('_', ' ').title()}** ({alt['score']:.1%})")
                            st.caption(alt.get('reason', ''))

                # Show content analysis insights
                with st.expander("Content Analysis Insights"):
                    analysis_data = rec.get('analysis', {})
                    if analysis_data:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Semantic Density", f"{analysis_data.get('semantic_density', 0):.2f}")
                            st.metric("Structural Complexity", f"{analysis_data.get('structural_complexity', 0):.2f}")
                            st.metric("Avg Sentence Length", f"{analysis_data.get('avg_sentence_length', 0):.1f} chars")
                        with col_b:
                            st.metric("Vocabulary Complexity", f"{analysis_data.get('vocabulary_complexity', 0):.2f}")
                            st.metric("Has Headers", "Yes" if analysis_data.get('has_headers') else "No")
                            st.metric("Code Blocks", analysis_data.get('code_blocks', 0))

            # Show metrics
            st.subheader("📊 Chunking Metrics")
            metrics = analysis['metrics']
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Overall Quality", f"{metrics.overall_quality:.3f}")
            with m_col2:
                st.metric("Coherence", f"{metrics.coherence_score:.3f}")
            with m_col3:
                st.metric("Boundary Quality", f"{metrics.boundary_quality:.3f}")

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    # Footer
    #st.markdown("---")
    #st.markdown("Built with Streamlit for intelligent RAG chunking strategy analysis.")

if __name__ == "__main__":
    main()