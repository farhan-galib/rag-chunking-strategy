"""
Evaluation metrics for chunking strategies
"""

from typing import List, Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class ChunkingMetrics:
    """Metrics for evaluating chunk quality"""
    coherence_score: float
    semantic_preservation: float
    retrieval_efficiency: float
    boundary_quality: float
    overall_quality: float

class ChunkEvaluator:
    """Evaluates the quality of chunks"""
    
    @staticmethod
    def calculate_coherence(chunks: List) -> float:
        """
        Calculate coherence score based on chunk sizes and uniformity
        Range: 0-1, higher is better
        """
        if not chunks:
            return 0.0
        
        sizes = [len(chunk.text) for chunk in chunks]
        
        # Calculate coefficient of variation
        mean_size = np.mean(sizes)
        std_dev = np.std(sizes)
        
        if mean_size == 0:
            return 0.0
        
        cv = std_dev / mean_size
        # Convert to 0-1 scale where 1 is perfect coherence
        coherence = max(0, 1 - cv)
        
        return round(coherence, 3)
    
    @staticmethod
    def calculate_semantic_preservation(chunk_text: str, original_text: str) -> float:
        """
        Calculate semantic preservation score
        Estimates how well meaning is preserved in chunks
        """
        # Simple heuristic: check for sentence boundaries
        sentences_in_chunk = chunk_text.count('. ')
        
        # Penalize if chunk ends abruptly (incomplete sentence)
        ends_with_period = chunk_text.rstrip().endswith('.')
        
        preservation_score = 0.8 if ends_with_period else 0.5
        
        return round(preservation_score, 3)
    
    @staticmethod
    def calculate_retrieval_efficiency(chunk_count: int, total_text_size: int) -> float:
        """
        Calculate retrieval efficiency
        Fewer chunks = faster retrieval, but may lose granularity
        """
        if total_text_size == 0:
            return 0.0
        
        avg_chunk_size = total_text_size / chunk_count if chunk_count > 0 else 0
        
        # Optimal chunk size is around 300-500 characters
        optimal_size = 400
        
        if avg_chunk_size == 0:
            return 0.0
        
        efficiency = 1 - (abs(avg_chunk_size - optimal_size) / optimal_size)
        efficiency = max(0, min(1, efficiency))
        
        return round(efficiency, 3)
    
    @staticmethod
    def calculate_boundary_quality(chunks: List) -> float:
        """
        Calculate quality of chunk boundaries
        Checks if chunks end at natural boundaries (sentences, paragraphs)
        """
        if not chunks:
            return 0.0
        
        good_boundaries = 0
        
        for chunk in chunks:
            text = chunk.text.rstrip()
            # Check for natural boundaries
            if (text.endswith('.') or text.endswith('!') or 
                text.endswith('?') or text.endswith('\n')):
                good_boundaries += 1
        
        quality = good_boundaries / len(chunks)
        return round(quality, 3)
    
    @staticmethod
    def evaluate_strategy(chunks: List, total_text_size: int) -> ChunkingMetrics:
        """Comprehensive evaluation of chunking strategy"""
        
        coherence = ChunkEvaluator.calculate_coherence(chunks)
        
        # Average semantic preservation
        semantic_scores = [ChunkEvaluator.calculate_semantic_preservation(c.text, "") 
                          for c in chunks]
        semantic_preservation = round(np.mean(semantic_scores), 3) if semantic_scores else 0.0
        
        retrieval_efficiency = ChunkEvaluator.calculate_retrieval_efficiency(
            len(chunks), total_text_size
        )
        
        boundary_quality = ChunkEvaluator.calculate_boundary_quality(chunks)
        
        # Weighted overall quality
        overall_quality = round(
            0.25 * coherence + 
            0.35 * semantic_preservation + 
            0.25 * retrieval_efficiency + 
            0.15 * boundary_quality,
            3
        )
        
        return ChunkingMetrics(
            coherence_score=coherence,
            semantic_preservation=semantic_preservation,
            retrieval_efficiency=retrieval_efficiency,
            boundary_quality=boundary_quality,
            overall_quality=overall_quality
        )

def print_evaluation_report(metrics: ChunkingMetrics, strategy_name: str):
    """Print a formatted evaluation report"""
    print(f"\n{'='*60}")
    print(f"Evaluation Report: {strategy_name}")
    print(f"{'='*60}")
    print(f"Coherence Score:              {metrics.coherence_score:.3f}")
    print(f"Semantic Preservation:        {metrics.semantic_preservation:.3f}")
    print(f"Retrieval Efficiency:         {metrics.retrieval_efficiency:.3f}")
    print(f"Boundary Quality:             {metrics.boundary_quality:.3f}")
    print(f"-" * 60)
    print(f"Overall Quality Score:        {metrics.overall_quality:.3f}")
    print(f"{'='*60}\n")