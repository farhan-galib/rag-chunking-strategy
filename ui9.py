"""
RAG Chunking Strategy UI
Web interface for document analysis and chunking recommendations
"""

import streamlit as st
import streamlit.components.v1 as components
import re
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
from file_extraction import (
    extract_text_from_file,
    get_supported_file_types,
    get_document_metadata,
    docx_to_html,
    load_excel_sheets,
)

def analyze_text(text: str) -> Dict:
    """Comprehensive text analysis with dynamic recommendations"""
    recommender = DynamicChunkingRecommender()
    recommendation_result = recommender.recommend_strategy(text)
    chunks = recommendation_result['strategy'].chunk(text, source='uploaded')
    metrics = ChunkEvaluator.evaluate_strategy(chunks, len(text))
    analysis = ChunkAnalyzer.analyze_chunks(chunks)

    return {
        'doc_type': recommendation_result['doc_type'],
        'recommendation': recommendation_result,
        'chunks': chunks,
        'metrics': metrics,
        'analysis': analysis,
        'text_length': len(text),
        'original_text': text
    }

def get_rag_readiness(overall_quality: float, confidence: float) -> str:
    """Return a simple readiness label for business users."""
    if overall_quality >= 0.75 and confidence >= 0.7:
        return "Deployment Ready"
    if overall_quality >= 0.6 and confidence >= 0.55:
        return "Caution"
    return "Manual Cleanup"

def _upload_signature(uploaded_files) -> Tuple:
    """Stable key to detect when the user changes uploaded files."""
    if not uploaded_files:
        return ()
    return tuple((f.name, f.size) for f in uploaded_files)

def process_uploaded_files(uploaded_files) -> List[Dict]:
    """Extract text from uploaded files and return preview-ready entries."""
    file_entries = []
    for uploaded_file in uploaded_files:
        try:
            file_ext = Path(uploaded_file.name).suffix.lower().strip(".")
            # Small fix: file_content should be read only once per file upload turn if possible
            # But Streamlit's file_uploader buffer is fine with .read() multiple times if handled
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            text, success = extract_text_from_file(file_content, file_ext)

            if success:
                metadata = get_document_metadata(
                    file_content, file_ext, uploaded_file.name
                )
                file_entries.append({
                    "name": uploaded_file.name,
                    "title": metadata["title"],
                    "text": text,
                    "extension": file_ext,
                    "size_bytes": metadata["size_bytes"],
                    "size_kb": metadata["size_kb"],
                    "page_count": metadata["page_count"],
                    "content": file_content,
                })
            else:
                st.error(f"❌ Error loading {uploaded_file.name}: {text}")
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    return file_entries

@st.dialog("📄 File Preview", width="large")
def file_preview_dialog(file_entries: List[Dict]) -> None:
    """Modal popup with metadata and extracted text for all uploaded files."""
    if not file_entries:
        st.info("No files to preview.")
        return

    tab_labels = [entry["title"] for entry in file_entries]
    tabs = st.tabs(tab_labels)
    for tab, entry in zip(tabs, file_entries):
        with tab:
            _render_single_file_preview(entry)

def render_visual_chunk_map(text: str, chunks: List[Chunk], limit_chunks: int = 20) -> str:
    """Renders a visual representation of how text is split into chunks."""
    if not chunks:
        return "<div style='color: #64748b; font-style: italic;'>No segments generated.</div>"
    
    # Colors for chunks (pastel-ish highlights)
    colors = [
        "rgba(99, 102, 241, 0.2)",  # Indigo
        "rgba(236, 72, 153, 0.2)",  # Pink
        "rgba(16, 185, 129, 0.2)",  # Emerald
        "rgba(245, 158, 11, 0.2)",  # Amber
        "rgba(59, 130, 246, 0.2)",  # Blue
        "rgba(139, 92, 246, 0.2)",  # Violet
    ]
    
    # We use a viewport of the text to keep it performant
    # Find the end of the last chunk we want to display
    display_chunks = chunks[:limit_chunks]
    max_pos = display_chunks[-1].end_char if display_chunks else 0
    viewport_text = text[:max_pos + 100] # Buffer
    
    html = '<div style="font-family: \'Courier New\', monospace; font-size: 11px; white-space: pre-wrap; background: #ffffff; padding: 20px; border: 1px solid #e2e8f0; border-radius: 12px; line-height: 1.8; max-height: 450px; overflow-y: auto; color: #1e293b; box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.03);">'
    
    last_idx = 0
    for i, chunk in enumerate(display_chunks):
        start = chunk.start_char
        end = chunk.end_char
        
        # Handle potential gaps or out-of-order issues (Safety)
        if start < last_idx: start = last_idx
        if end < start: end = start
        
        # Text before chunk (unmapped or gap)
        if start > last_idx:
            html += f'<span style="color: #cbd5e1;">{viewport_text[last_idx:start]}</span>'
            
        color = colors[i % len(colors)]
        # The mark itself
        html += f'<mark style="background-color: {color}; border-radius: 3px; padding: 2px 0; border: 1px solid rgba(0,0,0,0.05); cursor: help;" title="Segment {chunk.chunk_id} ({end-start} chars)">{viewport_text[start:end]}</mark>'
        last_idx = end
        
    if last_idx < len(text):
         html += f'<div style="text-align: center; margin-top: 15px; color: #94a3b8; font-style: italic; font-size: 10px;">... document continues ({len(text)-last_idx} more characters) ...</div>'
         
    html += '</div>'
    return html

@st.dialog("🔍 Analysis Results", width="large")
def analysis_results_dialog(analyses: List[Dict]):
    """Compact row-by-row presentation of analysis findings."""
    if not analyses:
        st.info("No analysis results to display.")
        return

    for analysis in analyses:
        name = analysis.get("filename", "Output")
        rec = analysis["recommendation"]
        metrics = analysis["metrics"]
        quality = metrics.overall_quality
        readiness = get_rag_readiness(quality, rec["confidence"])
        
        # Readiness colors for badges
        r_colors = {
            "Deployment Ready": {"bg": "#dcfce7", "text": "#166534", "border": "#22c55e"},
            "Caution": {"bg": "#fef9c3", "text": "#854d0e", "border": "#eab308"},
            "Manual Cleanup": {"bg": "#fee2e2", "text": "#991b1b", "border": "#ef4444"}
        }
        rc = r_colors.get(readiness, r_colors["Caution"])

        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h4 style="margin: 0;">📄 {name}</h4>
                <span style="background: {rc['bg']}; color: {rc['text']}; border: 1px solid {rc['border']}; 
                      padding: 4px 12px; border-radius: 20px; font-size: 10px; font-weight: 800; text-transform: uppercase;">
                    {readiness}
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="hero-card" style="padding: 24px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <div class="strategy-pill">{rec['strategy_name'].replace('_', ' ')}</div>
                        <div class="strategy-title" style="font-size: 24px; margin-bottom: 8px;">{rec['strategy_name'].replace('_', ' ').title()}</div>
                        <div class="reasoning-text" style="font-size: 14px; margin-top: 8px; max-width: 500px;">"{rec['reasoning']}"</div>
                    </div>
                    <div style="text-align: right;">
                        <div class="confidence-score" style="font-size: 36px;">{(rec['confidence'] * 100):.0f}%</div>
                        <div class="confidence-label">CONFIDENCE</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card" style="padding: 15px;"><div class="metric-value" style="font-size: 22px;">{metrics.overall_quality:.2f}</div><div class="metric-label">QUALITY</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card" style="padding: 15px;"><div class="metric-value" style="font-size: 22px;">{metrics.semantic_preservation:.2f}</div><div class="metric-label">SEMANTIC</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card" style="padding: 15px;"><div class="metric-value" style="font-size: 22px;">{metrics.boundary_quality:.2f}</div><div class="metric-label">BOUNDARY</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card" style="padding: 15px;"><div class="metric-value" style="font-size: 22px;">{metrics.retrieval_efficiency:.2f}</div><div class="metric-label">EFFICIENCY</div></div>', unsafe_allow_html=True)
        
        with st.expander("Alternative Strategies"):
            if rec.get('alternatives'):
                st.markdown("#### 🔄 Alternative Strategy Rankings")
                for alt in rec['alternatives'][:3]:
                    st.write(f"• **{alt['name'].replace('_', ' ').title()}** ({alt['score']:.1%})")
                    st.caption(alt.get('reason', ''))
                
                st.divider()
                st.markdown("#### 🧪 Strategy Playground")
                st.caption("Change settings below to preview how alternative strategies would handle this document.")
                
                strategy_map = {
                    "Fixed Size": FixedSizeChunking,
                    "Sliding Window": SlidingWindowChunking,
                    "Semantic": SemanticChunking,
                    "Recursive": RecursiveChunking,
                    "Hybrid": HybridChunking
                }
                
                # Use columns for playground controls
                pc1, pc2 = st.columns([1, 2])
                with pc1:
                    selected_strat_name = st.selectbox(
                        "Try Another Strategy",
                        options=list(strategy_map.keys()),
                        index=list(strategy_map.keys()).index(rec['strategy_name'].replace('_', ' ').title()) if rec['strategy_name'].replace('_', ' ').title() in strategy_map else 0,
                        key=_safe_widget_key(name, f"strat_select_{analysis.get('filename', 'out')}")
                    )
                
                with pc2:
                    st.info(f"Visualizing: **{selected_strat_name}**")

                # Get strategy class and chunk
                strat_class = strategy_map[selected_strat_name]
                simulation_text = analysis['original_text'][:15000] # Substantial sample
                
                try:
                    # Create temporary instance
                    temp_strat = strat_class()
                    temp_chunks = temp_strat.chunk(simulation_text)
                    
                    st.markdown(f"**Visual Segmentation Map** ({len(temp_chunks)} segments identified in sample)")
                    
                    # Display the visual map
                    st.markdown(render_visual_chunk_map(simulation_text, temp_chunks, limit_chunks=15), unsafe_allow_html=True)
                    
                    st.caption("💡 **Tip**: Hover over highlighted segments to see segment IDs and character lengths.")
                except Exception as e:
                    st.error(f"Error simulating strategy: {str(e)}")

            else:
                st.caption("No alternative strategies identified.")

        with st.expander("Technical Deep Dive"):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.write("**Relational Metrics**")
                doc_analysis = rec.get('analysis', {})
                st.write(f"- Semantic Density: {doc_analysis.get('semantic_density', 0):.2f}")
                st.write(f"- Vocabulary Complexity: {doc_analysis.get('vocabulary_complexity', 0):.2f}")
                st.write(f"- Coherence Score: {metrics.coherence_score:.2f}")
            with sc2:
                st.write("**Dataset Characteristics**")
                # Technical analysis results are stored in analysis['analysis']
                tech_data = analysis.get('analysis', {})
                st.write(f"- Total Chunks: {tech_data.get('total_chunks', '—')}")
                st.write(f"- Avg Chunk Size: {tech_data.get('avg_chunk_size', 0):.1f} chars")
                st.write(f"- Overall Quality Score: {metrics.overall_quality:.2f}")

        with st.expander("Sample Segments"):
            for idx, chunk in enumerate(analysis["chunks"][:3]):
                st.markdown(f"**Segment {idx+1}**")
                st.markdown(f"""
                    <div style="background: #f8fafc; padding: 12px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 11px; border: 1px solid #e2e8f0; color: #334155; line-height: 1.5;">
                        {chunk.text[:600]}...
                    </div>
                """, unsafe_allow_html=True)
                if idx < 2 and idx < len(analysis["chunks"]) - 1:
                    st.divider()

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()



@st.dialog("🎮 Strategy Playground", width="large")
def strategy_playground_dialog(analyses: List[Dict]):
    """Interactive workspace for experimenting with chunking strategies."""
    if not analyses:
        st.info("Please load or analyze a document first to enter the playground.")
        return

    st.markdown("### 🧪 Interactive Strategy Lab")
    st.markdown(
        "Experiment with different chunking models to see how they partition your data. "
        "The **AI Recommendation** is your baseline."
    )

    filenames = [a["filename"] for a in analyses]
    selected_filename = st.selectbox("Select Document", filenames)
    
    # Extract the analysis for the selected file
    current_analysis = next(a for a in analyses if a["filename"] == selected_filename)
    original_text = current_analysis.get("original_text", "")
    rec = current_analysis["recommendation"]

    # UI Panels
    c1, c2 = st.columns([1, 2.5])

    strategy_map = {
        "Semantic (AI)": SemanticChunking,
        "Recursive (Smart Context)": RecursiveChunking,
        "Hybrid (Optimized)": HybridChunking,
        "Sliding Window (Overlap)": SlidingWindowChunking,
        "Fixed Size (Predictable)": FixedSizeChunking,
    }

    strategy_guides = {
        "Semantic (AI)": {
            "description": "Uses machine learning or logic to detect topical boundaries rather than character counts.",
            "best_for": "Narrative documents, legal contracts, or philosophical texts where logical transitions are key.",
            "params": {
                "max_chunk_size": "Defines the physical limit for each segment to fit model context windows.",
                "min_chunk_size": "Prevents fragmented segments (like 'Cont.' or 'Note:') from losing their context."
            }
        },
        "Recursive (Smart Context)": {
            "description": "Splits by hierarchical structure: Headers first, then Paragraphs, then Sentences.",
            "best_for": "Technical docs, Markdown files, or academic papers with clear headers and sectioning.",
            "params": {
                "max_chunk_size": "The target length for sub-sections if a primary section is too long.",
                "chunk_overlap": "Ensures the end of one section flows naturally into the start of the next."
            }
        },
        "Hybrid (Optimized)": {
            "description": "A multi-pass approach that balances structural logic with performance efficiency.",
            "best_for": "Messy documents like OCR text, mixed business reports, or inconsistent layouts.",
            "params": {
                "semantic_weight": "Bias toward logical sentence boundaries vs. physical byte-size limits.",
                "window_weight": "Emphasis on overlapping context buffers for retrieval continuity."
            }
        },
        "Sliding Window (Overlap)": {
            "description": "Chunks by size but keeps a buffer to ensure ideas aren't severed mid-sentence.",
            "best_for": "Dense datasets (finance, medicine) where context from the prior sentence is vital.",
            "params": {
                "window_size": "The total physical length of each segment.",
                "step_size": "Controls the 'stride'. Smaller steps create higher redundancy/overlap."
            }
        },
        "Fixed Size (Predictable)": {
            "description": "Strict character-based partitioning. Extremely fast and predictable for scaling.",
            "best_for": "Massive datasets where retrieval speed is #1 priority and text is uniform.",
            "params": {
                "chunk_size": "The exact character count for each segment.",
                "overlap": "Safety buffer to prevent cutting words or short phrases in half."
            }
        }
    }

    with c1:
        st.markdown("#### ⚙️ Simulation Settings")
        
        # Determine current recommended strategy name for the dropdown default
        default_strat_label = next((k for k, v in strategy_map.items() if v.__name__ == rec['strategy'].__class__.__name__), list(strategy_map.keys())[0])

        selected_strat_label = st.selectbox(
            "Select Strategy",
            options=list(strategy_map.keys()),
            index=list(strategy_map.keys()).index(default_strat_label)
        )
        
        guide = strategy_guides[selected_strat_label]
        st.markdown(f"**Description**: *{guide['description']}*")
        
        # Context-aware recommendation highlighting
        current_doc_type = current_analysis.get('doc_type', 'General')
        is_recommended = rec['strategy_name'].replace('_', ' ').lower() in selected_strat_label.lower()
        
        if is_recommended:
            st.success(f"🎯 **AI Match**: Optimized for **{current_doc_type}** style content.")
        else:
            st.info(f"💡 **Best For**: {guide['best_for']}")

        st.divider()
        st.markdown("**🔧 Tune Parameters**")
        
        # Dynamic sliders based on selected strategy
        params = {}
        if selected_strat_label == "Fixed Size (Predictable)":
            params['chunk_size'] = st.slider("Chunk Size", 100, 2000, 500, help=guide['params']['chunk_size'])
            params['overlap'] = st.slider("Overlap", 0, 500, 50, help=guide['params']['overlap'])
        elif selected_strat_label == "Sliding Window (Overlap)":
            params['window_size'] = st.slider("Window Size", 100, 2000, 500, help=guide['params']['window_size'])
            params['step_size'] = st.slider("Step Size", 50, 1000, 250, help=guide['params']['step_size'])
        elif selected_strat_label == "Semantic (AI)":
            params['max_chunk_size'] = st.slider("Max Size", 200, 2000, 600, help=guide['params']['max_chunk_size'])
            params['min_chunk_size'] = st.slider("Min Size", 20, 500, 100, help=guide['params']['min_chunk_size'])
        elif selected_strat_label == "Recursive (Smart Context)":
            params['max_chunk_size'] = st.slider("Max Section Size", 500, 3000, 1000, help=guide['params']['max_chunk_size'])
            params['chunk_overlap'] = st.slider("Hierarchy Overlap", 0, 500, 100, help=guide['params']['chunk_overlap'])
        elif selected_strat_label == "Hybrid (Optimized)":
            params['semantic_weight'] = st.slider("Semantic Logic Bias", 0.0, 1.0, 0.6, help=guide['params']['semantic_weight'])
            params['window_weight'] = st.slider("Physical Limit Bias", 0.0, 1.0, 0.4, help=guide['params']['window_weight'])

        st.divider()
        st.markdown("**Performance Prediction**")
        
        # Simulation Logic
        strat_class = strategy_map[selected_strat_label]
        simulation_text = original_text[:20000] # Substantial slice for playground
        
        try:
            with st.spinner("Simulating..."):
                temp_strat = strat_class(**params)
                sim_chunks = temp_strat.chunk(simulation_text)
                sim_metrics = ChunkEvaluator.evaluate_strategy(sim_chunks, len(simulation_text))
                
                # Metric Cards for Playground
                def _mini_metric(label, value):
                    color = "#22c55e" if value >= 0.75 else "#eab308" if value >= 0.6 else "#ef4444"
                    st.markdown(f"""
                        <div style="background: white; border: 1px solid #e2e8f0; border-left: 4px solid {color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                            <div style="font-size: 10px; color: #64748b; font-weight: 800; text-transform: uppercase;">{label}</div>
                            <div style="font-size: 18px; font-weight: 800; color: {color};">{value:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)

                _mini_metric("Avg Quality", sim_metrics.overall_quality)
                _mini_metric("Semantic Preservation", sim_metrics.semantic_preservation)
                _mini_metric("Boundary Quality", sim_metrics.boundary_quality)
                
                st.caption(f"Estimated Retrieval Speed: **{'High' if sim_metrics.retrieval_efficiency > 0.8 else 'Medium'}**")

        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")

    with c2:
        st.markdown("#### 🗺️ Segmentation Map")
        st.markdown("This map shows the first **15 segments** generated by the selected strategy.")
        
        if 'sim_chunks' in locals():
            st.markdown(render_visual_chunk_map(simulation_text, sim_chunks, limit_chunks=15), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📑 Segment Inspector")
            tab1, tab2 = st.tabs(["Raw Text Preview", "Segment Metadata"])
            
            with tab1:
                for i, chunk in enumerate(sim_chunks[:3]):
                    st.markdown(f"**Segment {i+1}**")
                    st.markdown(f"""
                        <div style="background: #f8fafc; padding: 12px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 11px; border: 1px solid #e2e8f0; color: #334155;">
                            {chunk.text[:500]}...
                        </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                df_meta = pd.DataFrame([{
                    "ID": c.chunk_id,
                    "Chars": c.end_char - c.start_char,
                    "Start": c.start_char,
                    "End": c.end_char
                } for c in sim_chunks[:10]])
                st.dataframe(df_meta, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 🤖 RAG Recommendation")
    if rec['strategy'].__class__.__name__ == strat_class.__name__:
        st.success(f"**This is the AI Recommended strategy for this document!** {rec['reasoning']}")
    else:
        st.warning(f"**Note**: The AI prefers **{rec['strategy_name'].replace('_', ' ').title()}** for this document. Reasoning: *{rec['reasoning']}*")


@st.dialog("📊 Executive Analytics", width="large")
def analytics_dashboard_dialog(analyses: List[Dict]):
    """Business-friendly dashboard with charts and simple summaries."""
    if not analyses:
        st.info("No analytics available. Run 'Analyze' first.")
        return

    # --- Header Scorecard ---
    st.markdown("### 🏆 Portfolio Overview")
    t1, t2, t3, t4 = st.columns(4)
    
    ready_count = sum(1 for a in analyses if get_rag_readiness(a['metrics'].overall_quality, a['recommendation']['confidence']) == "Deployment Ready")
    caution_count = sum(1 for a in analyses if get_rag_readiness(a['metrics'].overall_quality, a['recommendation']['confidence']) == "Caution")
    cleanup_count = sum(1 for a in analyses if get_rag_readiness(a['metrics'].overall_quality, a['recommendation']['confidence']) == "Manual Cleanup")
    avg_quality = sum(a['metrics'].overall_quality for a in analyses) / len(analyses)

    t1.markdown(f'<div class="metric-card" style="border-top: 4px solid #22c55e;"><div class="metric-value" style="color: #16a34a;">{ready_count}</div><div class="metric-label">READY</div></div>', unsafe_allow_html=True)
    t2.markdown(f'<div class="metric-card" style="border-top: 4px solid #eab308;"><div class="metric-value" style="color: #ca8a04;">{caution_count}</div><div class="metric-label">CAUTION</div></div>', unsafe_allow_html=True)
    t3.markdown(f'<div class="metric-card" style="border-top: 4px solid #ef4444;"><div class="metric-value" style="color: #dc2626;">{cleanup_count}</div><div class="metric-label">CLEANUP</div></div>', unsafe_allow_html=True)
    t4.markdown(f'<div class="metric-card" style="border-top: 4px solid #6366f1;"><div class="metric-value" style="color: #4f46e5;">{avg_quality:.2f}</div><div class="metric-label">AVG QUALITY</div></div>', unsafe_allow_html=True)

    st.divider()

    # --- Visual Infographics ---
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("🕸️ Multi-Metric Quality Analysis")
        # Radar Chart for metrics comparison
        categories = ['Quality', 'Semantic', 'Boundary', 'Efficiency', 'Coherence']
        fig = go.Figure()

        for a in analyses[:5]: # Limit to first 5 for readability
            m = a['metrics']
            fig.add_trace(go.Scatterpolar(
                r=[m.overall_quality, m.semantic_preservation, m.boundary_quality, m.retrieval_efficiency, m.coherence_score],
                theta=categories,
                fill='toself',
                name=a['filename'][:15]
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=400,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🧩 Strategy Distribution")
        # Donut chart for strategies
        strategy_counts = Counter([a['recommendation']['strategy_name'].replace('_', ' ').title() for a in analyses])
        df_strat = pd.DataFrame(strategy_counts.items(), columns=['Strategy', 'Count'])
        fig_pie = px.pie(df_strat, values='Count', names='Strategy', hole=0.6,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    
    # --- Performance Benchmarks ---
    st.subheader("📊 Document Efficiency vs. Quality")
    plot_df = pd.DataFrame([{
        "Document": a["filename"],
        "Quality": a["metrics"].overall_quality,
        "Efficiency": a["metrics"].retrieval_efficiency,
        "Chunks": a.get('analysis', {}).get('total_chunks', 0),
        "Size": a["text_length"]
    } for a in analyses])
    
    fig_scatter = px.scatter(plot_df, x="Efficiency", y="Quality", size="Chunks", hover_name="Document",
                             color="Quality", color_continuous_scale="Viridis", height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    
    st.subheader("💡 Business Summaries")
    
    cols = st.columns(2)
    for idx, analysis in enumerate(analyses):
        with cols[idx % 2]:
            name = analysis["filename"]
            rec = analysis["recommendation"]
            quality = analysis["metrics"].overall_quality
            readiness = get_rag_readiness(quality, rec["confidence"])
            
            # Specific label colors
            label_color = "#22c55e" if readiness == "Deployment Ready" else "#eab308" if readiness == "Caution" else "#ef4444"
            bg_color = "#dcfce7" if readiness == "Deployment Ready" else "#fef9c3" if readiness == "Caution" else "#fee2e2"
            text_color = "#166534" if readiness == "Deployment Ready" else "#854d0e" if readiness == "Caution" else "#991b1b"

            # Business interpretation logic
            if readiness == "Deployment Ready":
                impact = "✅ **Deployment Ready**: This document is perfectly optimized. AI will find information accurately and quickly."
                action = "No action needed. Ready for ingestion."
            elif readiness == "Caution":
                impact = "⚠️ **Caution Advised**: Structural irregularities found. Retrieval may occasionally return irrelevant context."
                action = "Recommendation: Review tables and complex formatting before final deployment."
            else:
                impact = "🛑 **Manual Cleanup Required**: High noise levels or excessive length. Significant risk of AI hallucinations."
                action = "Action: Break into separate files or use heavy semantic filtering."

            st.markdown(f"""
                <div style="background: #ffffff; padding: 24px; border-radius: 16px; border-left: 6px solid {label_color}; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 24px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-weight: 800; color: #0f172a; font-size: 16px;">{name}</span>
                        <span style="background: {bg_color}; color: {text_color}; padding: 4px 12px; border-radius: 20px; font-size: 10px; font-weight: 900; text-transform: uppercase; letter-spacing: 0.5px;">
                            {readiness}
                        </span>
                    </div>
                    <div style="font-size: 14px; line-height: 1.5; color: #334155; margin-bottom: 16px;">
                        {impact}
                    </div>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px dashed #cbd5e1;">
                         <p style="font-size: 12px; color: #475569; margin: 0;"><strong>Next Step:</strong> {action}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def run_upload_analysis(file_entries: List[Dict]) -> List[Dict]:
    """Analyze all uploaded files and attach filenames to results."""
    analyses = []
    for entry in file_entries:
        analysis = analyze_text(entry["text"])
        analysis["filename"] = entry["name"]
        analyses.append(analysis)
    return analyses

def _page_count_label(ext: str) -> str:
    if ext.lower() in ("xlsx", "xls", "xl"):
        return "Sheets"
    return "Pages"

def _safe_widget_key(name: str, prefix: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    return f"{prefix}_{safe}"

def _render_pdf_preview(content: bytes, file_key: str, fallback_text: str = "") -> None:
    """IFrame embed for PDF preview."""
    import base64
    base64_pdf = base64.b64encode(content).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def _render_formatted_preview(entry: Dict) -> None:
    ext = entry.get("extension", "").lower()
    content = entry.get("content", b"")
    text = entry.get("text", "")
    name = entry["name"]

    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="content-header">📄 {name} Content</div>', unsafe_allow_html=True)

    if ext == "pdf" and content:
        _render_pdf_preview(content, file_key=name, fallback_text=text)
    elif ext in ("docx", "doc") and content:
        try:
            html = docx_to_html(content)
            st.markdown(f"<div class='scrollable-content'>{html}</div>", unsafe_allow_html=True)
        except Exception:
            st.markdown(f"<div class='scrollable-content'>{text}</div>", unsafe_allow_html=True)
    elif ext in ("xlsx", "xls", "xl") and content:
        try:
            workbook = load_excel_sheets(content, ext)
            sheet_names = workbook["sheet_names"]
            sheets = workbook["sheets"]
            st.markdown('<div style="padding: 16px;">', unsafe_allow_html=True)
            if len(sheet_names) == 1:
                st.dataframe(sheets[sheet_names[0]], use_container_width=True)
            else:
                sheet_tabs = st.tabs(sheet_names)
                for tab, sheet_name in zip(sheet_tabs, sheet_names):
                    with tab:
                        st.dataframe(sheets[sheet_name], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown(f"<div class='scrollable-content'>{text}</div>", unsafe_allow_html=True)
    elif ext == "md":
        st.markdown(f"<div class='scrollable-content'>{text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='scrollable-content'>{text}</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def _render_single_file_preview(entry: Dict) -> None:
    name = entry["name"]
    ext = entry.get("extension", "unknown").upper()
    size_kb = entry.get("size_kb", "—")
    page_count = entry.get("page_count", 1)
    page_label = _page_count_label(entry.get("extension", ""))

    st.markdown(f"""
        <div class="preview-grid">
            <div class="preview-meta-item">
                <div class="stat-label">File Format</div>
                <div class="stat-value">{ext}</div>
            </div>
            <div class="preview-meta-item">
                <div class="stat-label">File Size</div>
                <div class="stat-value">{size_kb}</div>
            </div>
            <div class="preview-meta-item">
                <div class="stat-label">{page_label}</div>
                <div class="stat-value">{page_count}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    _render_formatted_preview(entry)

def render_business_dashboard(analyses: List[Dict]):
    st.header("📈 RAG Readiness Dashboard")
    if not analyses:
        st.info("Upload documents and run analysis to see insights.")
        return

    total_docs = len(analyses)
    avg_quality = sum(a['metrics'].overall_quality for a in analyses) / total_docs
    avg_confidence = sum(a['recommendation']['confidence'] for a in analyses) / total_docs
    
    st.markdown("### 🏆 Portfolio Performance")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{total_docs}</div><div class="metric-label">PROCESSED DOCUMENTS</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{avg_quality:.2f}</div><div class="metric-label">AVG RAG QUALITY</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{avg_confidence:.1%}</div><div class="metric-label">AVG CONFIDENCE</div></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💡 Document Summaries")
    
    cols = st.columns(2)
    for idx, analysis in enumerate(analyses):
        with cols[idx % 2]:
            name = analysis.get('filename', 'Unknown')
            rec = analysis['recommendation']
            quality = analysis['metrics'].overall_quality
            readiness = get_rag_readiness(quality, rec['confidence'])
            
            label_color = "#22c55e" if readiness == "Deployment Ready" else "#eab308" if readiness == "Caution" else "#ef4444"
            bg_color = "#dcfce7" if readiness == "Deployment Ready" else "#fef9c3" if readiness == "Caution" else "#fee2e2"
            text_color = "#166534" if readiness == "Deployment Ready" else "#854d0e" if readiness == "Caution" else "#991b1b"

            st.markdown(f"""
                <div style="background: #ffffff; padding: 20px; border-radius: 16px; border-left: 6px solid {label_color}; box-shadow: 0 4px 10px rgba(0,0,0,0.03); margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-weight: 700; color: #1e293b;">{name}</span>
                        <span style="background: {bg_color}; color: {text_color}; padding: 2px 10px; border-radius: 20px; font-size: 10px; font-weight: 800; text-transform: uppercase;">
                            {readiness}
                        </span>
                    </div>
                    <div style="font-size: 13px; color: #475569; margin-bottom: 8px;">
                        <strong>Recommendation:</strong> {rec['strategy_name'].replace('_', ' ').title()}
                    </div>
                    <div style="font-size: 12px; color: #64748b;">
                        {rec['reasoning'][:120]}...
                    </div>
                </div>
            """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="RAG Chunking Strategy Advisor", page_icon="📄", layout="wide")

    # --- Screenshot Design CSS ---
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
        
        .hero-card {
            background-color: white;
            border-radius: 20px;
            padding: 40px;
            border-left: 8px solid #6366f1;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
            margin-bottom: 40px;
            font-family: 'Inter', sans-serif;
        }
        
        .strategy-pill {
            background-color: #eef2ff;
            color: #4338ca;
            padding: 4px 12px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            display: inline-block;
            margin-bottom: 24px;
        }
        
        .strategy-title {
            font-family: 'Outfit', sans-serif;
            font-size: 36px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 24px;
        }
        
        .confidence-score {
            font-family: 'Outfit', sans-serif;
            font-size: 44px;
            font-weight: 800;
            color: #6366f1;
            display: inline-block;
        }
        
        .confidence-label {
            font-family: 'Outfit', sans-serif;
            font-size: 14px;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            margin-left: 10px;
            letter-spacing: 1px;
        }
        
        .reasoning-text {
            font-style: italic;
            color: #475569;
            font-size: 18px;
            line-height: 1.6;
            margin-top: 24px;
        }
        
        .metrics-title {
            font-family: 'Outfit', sans-serif;
            font-size: 28px;
            font-weight: 800;
            color: #1a202c;
            margin-bottom: 24px;
            margin-top: 40px;
        }
        
        .metric-card {
            background-color: white;
            border-radius: 20px;
            padding: 30px 20px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            border: 1px solid #f1f5f9;
        }
        
        .metric-value {
            font-family: 'Outfit', sans-serif;
            font-size: 32px;
            font-weight: 800;
            color: #6366f1;
            margin-bottom: 8px;
        }
        
        .metric-label {
            font-family: 'Outfit', sans-serif;
            font-size: 11px;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* --- Action Bar & Button Styling --- */
        .stButton > button {
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 11px;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background-color: white;
            color: #1e293b;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            width: 100%;
        }

        .stButton > button:hover {
            border-color: #6366f1;
            color: #6366f1;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.15);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
            color: white;
            box-shadow: 0 12px 24px -8px rgba(79, 70, 229, 0.4);
        }

        /* Target the action container row */
        [data-testid="stHorizontalBlock"]:has(button) {
            background: #f8fafc;
            padding: 24px;
            border-radius: 24px;
            border: 1px solid #e2e8f0;
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.03);
            align-items: center;
        }
        
        /* --- Compact Preview Styling --- */
        .preview-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
        }
        
        .preview-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 24px;
        }
        
        .preview-meta-item {
            background-color: white;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #6366f1;
            border-radius: 12px;
            padding: 16px 20px;
            text-align: left;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }
        
        .stat-label {
            font-size: 11px;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 6px;
        }
        
        .stat-value {
            font-family: 'Outfit', sans-serif;
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
        }
        
        .content-container {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        
        .content-header {
            background: #f1f5f9;
            padding: 8px 16px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 600;
            color: #475569;
        }

        .scrollable-content {
            padding: 24px;
            max-height: 500px;
            overflow-y: auto;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            line-height: 1.6;
            color: #334155;
            background: #fafafa;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📄 RAG Chunking Analyzer")
    
    st.markdown(f"""
        <div class="hero-card" style="padding: 30px; border-bottom: 4px solid #6366f1;">
            <div class="strategy-pill">Enterprise AI Optimization</div>
            <div class="strategy-title" style="font-size: 32px; margin-bottom: 12px;">Document IQ & Chunking Strategy Advisor</div>
            <p style="color: #475569; font-size: 16px; margin: 0;">
                Analyze complex enterprise documents to discover the mathematical ideal for chunking. 
                Improve retrieval accuracy, reduce hallucinations, and optimize your RAG pipelines.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("View:", ["Upload & Analyze", "Business Dashboard"])

    st.sidebar.header("📚 Built-in Datasets")
    available_datasets = list_datasets()
    selected_dataset = st.sidebar.selectbox(
        "Built-in dataset:",
        ["None"] + available_datasets,
        key="dataset_select"
    )

    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, XLSX, TXT, MD)",
        type=get_supported_file_types(),
        accept_multiple_files=True,
        key="uploader",
    )

    if "file_entries" not in st.session_state:
        st.session_state.file_entries = []
    if "upload_analyses" not in st.session_state:
        st.session_state.upload_analyses = []
    if "analyze_requested" not in st.session_state:
        st.session_state.analyze_requested = False
    
    current_sig = _upload_signature(uploaded_files)
    if "sig" not in st.session_state or st.session_state.sig != current_sig:
        st.session_state.sig = current_sig
        st.session_state.file_entries = process_uploaded_files(uploaded_files) if uploaded_files else []
        st.session_state.upload_analyses = []
        st.session_state.analyze_requested = False

    file_entries = st.session_state.file_entries
    
    dataset_analysis = None
    if selected_dataset != "None":
        dataset_analysis = analyze_text(load_dataset(selected_dataset))
        dataset_analysis["filename"] = selected_dataset

    all_analyses = (st.session_state.upload_analyses if st.session_state.analyze_requested else []) + ([dataset_analysis] if dataset_analysis else [])

    if page == "Upload & Analyze":
        # Upload View Flow
        if file_entries:
            st.markdown("### 📋 Ready for Analysis")
            
            # Show file summaries in the grid
            for i in range(0, len(file_entries), 3):
                row = file_entries[i:i+3]
                grid_cols = st.columns(3)
                for entry, g_col in zip(row, grid_cols):
                    with g_col:
                        ext = entry.get("extension", "unknown").upper()
                        size = entry.get("size_kb", "—")
                        pages = entry.get("page_count", 1)
                        unit_label = _page_count_label(ext).upper()
                        st.markdown(f"""
                            <div class="preview-meta-item" style="margin-bottom: 12px;">
                                <div style="font-weight: 700; font-size: 14px; margin-bottom: 8px; color: #0f172a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                    📄 {entry['name']}
                                </div>
                                <div style="display: flex; gap: 10px; font-size: 10px; color: #64748b;">
                                    <span>TEXT</span> <span style="color: #6366f1; font-weight: 800;">{ext}</span>
                                    <span>SIZE</span> <span style="color: #6366f1; font-weight: 800;">{size}</span>
                                    <span>{unit_label}</span> <span style="color: #6366f1; font-weight: 800;">{pages}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            # --- Centered Action Bar ---
            st.markdown("<br>", unsafe_allow_html=True)
            _, c1, c2, c3, c4, _ = st.columns([0.5, 1, 1, 1, 1, 0.5])
            with c1:
                if st.button("👁️ Preview", use_container_width=True):
                    file_preview_dialog(file_entries)
            with c2:
                if st.button("🔍 Analyze", type="primary", use_container_width=True):
                    with st.spinner("Decoding document patterns..."):
                        analyses = run_upload_analysis(file_entries)
                        st.session_state.upload_analyses = analyses
                        st.session_state.analyze_requested = True
                        analysis_results_dialog(analyses)
            with c3:
                has_results = bool(st.session_state.upload_analyses or dataset_analysis)
                if st.button("📊 Analytics", use_container_width=True, disabled=not has_results):
                    analytics_dashboard_dialog(all_analyses)
            with c4:
                if st.button("🎮 Playground", use_container_width=True, disabled=not has_results):
                    strategy_playground_dialog(all_analyses)

        elif selected_dataset != "None":
            st.markdown(f"""
                <div class="hero-card" style="padding: 20px; border-left: 8px solid #94a3b8; background-color: #f1f5f9;">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <div style="font-size: 40px;">📚</div>
                        <div>
                            <div style="font-weight: 800; font-size: 18px; color: #1e293b;">{selected_dataset}</div>
                            <div style="font-size: 13px; color: #475569;">Internal reference dataset selected. Performance will be benchmarked against standard RAG patterns.</div>
                        </div>
                    </div>
                </div>
                <br>
            """, unsafe_allow_html=True)
            
            # --- Centered Action Bar for Dataset ---
            st.markdown("<br>", unsafe_allow_html=True)
            _, c1, c2, c3, _ = st.columns([1, 1.2, 1.2, 1.2, 1])
            with c1:
                if st.button("🔍 Analyze Dataset", type="primary", use_container_width=True):
                    analysis_results_dialog([dataset_analysis])
            with c2:
                if st.button("📊 Analytics", use_container_width=True):
                    analytics_dashboard_dialog([dataset_analysis])
            with c3:
                if st.button("🎮 Playground", use_container_width=True):
                    strategy_playground_dialog([dataset_analysis])
        else:
            st.markdown("""
                <div style="text-align: center; padding: 60px; background: #f8fafc; border: 2px dashed #e2e8f0; border-radius: 20px;">
                    <div style="font-size: 48px; margin-bottom: 20px;">📂</div>
                    <div style="font-weight: 700; color: #475569; font-size: 18px;">No Data Loaded</div>
                    <p style="color: #64748b; font-size: 14px;">Upload your proprietary documents or select a built-in dataset to begin RAG optimization.</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        render_business_dashboard(all_analyses)

if __name__ == "__main__":
    main()
