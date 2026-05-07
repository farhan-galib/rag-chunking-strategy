"""
Main RAG Chunking Strategy Demo
Demonstrates all chunking strategies with various datasets
"""

from chunking_strategies import (
    FixedSizeChunking, SlidingWindowChunking, SemanticChunking,
    RecursiveChunking, HybridChunking, ChunkAnalyzer
)
from datasets import DATASETS
from evaluation import ChunkEvaluator, print_evaluation_report

def demonstrate_all_strategies():
    """Demonstrate all chunking strategies"""
    
    # Select a dataset
    dataset_name = 'technical_faq'
    text = DATASETS[dataset_name]
    
    print(f"\n{'='*60}")
    print(f"RAG Chunking Strategy Demonstration")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Text Length: {len(text)} characters")
    print(f"{'='*60}\n")
    
    # Define strategies to test
    strategies = {
        'Fixed Size (500 chars, no overlap)': FixedSizeChunking(chunk_size=500, overlap=0),
        'Sliding Window (500 chars, 250 step)': SlidingWindowChunking(window_size=500, step_size=250),
        'Semantic Chunking': SemanticChunking(max_chunk_size=500),
        'Recursive/Hierarchical': RecursiveChunking(max_chunk_size=1000),
        'Hybrid Strategy': HybridChunking(),
    }
    
    # Apply each strategy and evaluate
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\nApplying: {strategy_name}")
        print("-" * 60)
        
        # Chunk the text
        chunks = strategy.chunk(text, source=dataset_name)
        
        # Analyze chunks
        analysis = ChunkAnalyzer.analyze_chunks(chunks)
        print(f"Total Chunks: {analysis['total_chunks']}")
        print(f"Avg Chunk Size: {analysis['avg_chunk_size']:.1f} chars")
        print(f"Min/Max: {analysis['min_chunk_size']} / {analysis['max_chunk_size']} chars")
        print(f"Std Dev: {analysis['std_dev']:.1f}")
        
        # Evaluate strategy
        metrics = ChunkEvaluator.evaluate_strategy(chunks, len(text))
        results[strategy_name] = {
            'chunks': chunks,
            'analysis': analysis,
            'metrics': metrics
        }
        
        print_evaluation_report(metrics, strategy_name)
        
        # Show sample chunks
        print("Sample Chunks:")
        ChunkAnalyzer.print_chunks(chunks[:2], max_preview=150)
    
    # Print comparison
    print_comparison_table(results)
    
    return results

def print_comparison_table(results):
    """Print comparison table of all strategies"""
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Strategy':<35} {'Chunks':<10} {'Quality':<10} {'Coherence':<10} {'Boundary':<10}")
    print(f"{'-'*80}")
    
    for strategy_name, data in results.items():
        metrics = data['metrics']
        chunks_count = data['analysis']['total_chunks']
        print(f"{strategy_name:<35} {chunks_count:<10} {metrics.overall_quality:<10.3f} "
              f"{metrics.coherence_score:<10.3f} {metrics.boundary_quality:<10.3f}")
    
    print(f"{'='*80}\n")

def test_with_all_datasets():
    """Test all strategies with all datasets"""
    print(f"\n{'='*80}")
    print("TESTING ALL STRATEGIES WITH ALL DATASETS")
    print(f"{'='*80}\n")
    
    strategies = {
        'Fixed Size': FixedSizeChunking(chunk_size=500),
        'Semantic': SemanticChunking(),
        'Hybrid': HybridChunking(),
    }
    
    for dataset_name, text in DATASETS.items():
        print(f"\nDataset: {dataset_name.upper()}")
        print("-" * 60)
        
        for strategy_name, strategy in strategies.items():
            chunks = strategy.chunk(text, source=dataset_name)
            metrics = ChunkEvaluator.evaluate_strategy(chunks, len(text))
            
            print(f"  {strategy_name:<20} | Chunks: {len(chunks):<3} | "
                  f"Quality: {metrics.overall_quality:.3f}")

if __name__ == "__main__":
    # Run main demonstration
    results = demonstrate_all_strategies()
    
    # Test with all datasets
    test_with_all_datasets()
    
    print("\n✅ Demonstration complete!")