"""
Dynamic Chunking Strategy Recommender
Intelligent recommendation system that adapts chunking methods based on document characteristics
"""

import re
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict

from chunking_strategies import (
    FixedSizeChunking, SlidingWindowChunking, SemanticChunking,
    RecursiveChunking, HybridChunking, Chunk
)
from evaluation import ChunkEvaluator

class DynamicChunkingRecommender:
    """
    Dynamic recommender that analyzes document characteristics
    and recommends the optimal chunking strategy
    """

    def __init__(self):
        self.strategies = {
            'fixed_size': FixedSizeChunking(chunk_size=400, overlap=0),
            'sliding_window': SlidingWindowChunking(window_size=400, step_size=200),
            'semantic': SemanticChunking(max_chunk_size=400),
            'recursive': RecursiveChunking(max_chunk_size=800),
            'hybrid': HybridChunking()
        }

        # Strategy performance weights for different scenarios
        self.strategy_weights = {
            'fixed_size': {
                'code': 0.8, 'support': 0.7, 'general': 0.6,
                'technical': 0.5, 'legal': 0.4, 'medical': 0.4, 'ecommerce': 0.5
            },
            'sliding_window': {
                'technical': 0.7, 'medical': 0.6, 'general': 0.5,
                'code': 0.4, 'legal': 0.4, 'support': 0.4, 'ecommerce': 0.5
            },
            'semantic': {
                'technical': 0.9, 'medical': 0.8, 'legal': 0.7, 'general': 0.7,
                'ecommerce': 0.6, 'code': 0.5, 'support': 0.5
            },
            'recursive': {
                'legal': 0.9, 'technical': 0.7, 'code': 0.6, 'ecommerce': 0.6,
                'medical': 0.5, 'general': 0.5, 'support': 0.4
            },
            'hybrid': {
                'ecommerce': 0.8, 'technical': 0.7, 'general': 0.6,
                'medical': 0.6, 'legal': 0.5, 'code': 0.5, 'support': 0.5
            }
        }

    def recommend_strategy(self, text: str, doc_type: str = None) -> Dict:
        """
        Dynamic recommendation based on multiple factors

        Args:
            text: The document text to analyze
            doc_type: Optional document type override

        Returns:
            Dictionary with recommendation details
        """

        # Analyze text characteristics
        analysis = self._analyze_text_characteristics(text)

        # Auto-detect document type if not provided
        if not doc_type:
            doc_type = self._detect_document_type(text)

        # Score each strategy
        scores = {}
        for name, strategy in self.strategies.items():
            scores[name] = self._score_strategy(name, strategy, analysis, doc_type)

        # Get top 3 recommendations
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_strategy_name, best_score = sorted_strategies[0]

        # Optimize parameters for the best strategy
        optimized_strategy = self._optimize_strategy_parameters(
            self.strategies[best_strategy_name], analysis
        )

        return {
            'strategy': optimized_strategy,
            'strategy_name': best_strategy_name,
            'confidence': best_score,
            'reasoning': self._generate_reasoning(best_strategy_name, analysis, doc_type),
            'alternatives': [
                {
                    'name': name,
                    'score': score,
                    'reason': self._generate_reasoning(name, analysis, doc_type)
                } for name, score in sorted_strategies[1:4]  # Top 3 alternatives
            ],
            'analysis': analysis,
            'doc_type': doc_type
        }

    def _analyze_text_characteristics(self, text: str) -> Dict:
        """Analyze various text characteristics for better recommendations"""

        # For efficiency with large documents, analyze a sample
        if len(text) > 50000:
            sample = self._get_representative_sample(text)
            text = sample

        return {
            'length': len(text),
            'avg_sentence_length': self._calculate_avg_sentence_length(text),
            'structural_complexity': self._assess_structural_complexity(text),
            'semantic_density': self._calculate_semantic_density(text),
            'has_headers': bool(re.search(r'^#{1,6}\s', text, re.MULTILINE)),
            'has_lists': bool(re.search(r'^[\s]*[-\*\+]|\d+\.', text, re.MULTILINE)),
            'code_blocks': len(re.findall(r'```', text)),
            'avg_paragraph_length': self._calculate_avg_paragraph_length(text),
            'vocabulary_complexity': self._assess_vocabulary_complexity(text),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'paragraph_count': len(re.split(r'\n\s*\n', text.strip()))
        }

    def _get_representative_sample(self, text: str, sample_size: int = 10000) -> str:
        """Get a representative sample from large documents"""
        total_len = len(text)

        # Take samples from beginning, middle, and end
        part_size = sample_size // 3

        beginning = text[:part_size]
        middle_start = total_len // 2 - part_size // 2
        middle = text[middle_start:middle_start + part_size]
        end_start = total_len - part_size
        end = text[end_start:]

        return beginning + middle + end

    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in characters"""
        sentences = re.findall(r'[^.!?]+[.!?]+', text)
        if not sentences:
            return 0
        return np.mean([len(s.strip()) for s in sentences])

    def _assess_structural_complexity(self, text: str) -> float:
        """Assess how structurally complex the document is (0-1 scale)"""
        complexity = 0.0

        # Headers increase complexity
        header_count = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
        if header_count > 0:
            complexity += min(0.4, header_count * 0.05)

        # Lists increase complexity
        list_count = len(re.findall(r'^[\s]*[-\*\+]|\d+\.', text, re.MULTILINE))
        if list_count > 0:
            complexity += min(0.3, list_count * 0.02)

        # Code blocks increase complexity
        code_blocks = len(re.findall(r'```', text))
        if code_blocks > 0:
            complexity += min(0.3, code_blocks * 0.1)

        return min(1.0, complexity)

    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density based on technical terms and complexity"""
        # Simple heuristic: ratio of long words to total words
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0

        long_words = [w for w in words if len(w) > 6]
        density = len(long_words) / len(words)

        # Technical indicators
        technical_terms = ['algorithm', 'implementation', 'function', 'method', 'class',
                          'interface', 'protocol', 'framework', 'library', 'api']

        technical_count = sum(1 for term in technical_terms if term in text.lower())
        technical_density = min(1.0, technical_count / 10)

        return min(1.0, (density + technical_density) / 2)

    def _calculate_avg_paragraph_length(self, text: str) -> float:
        """Calculate average paragraph length"""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return 0

        return np.mean([len(p) for p in paragraphs])

    def _assess_vocabulary_complexity(self, text: str) -> float:
        """Assess vocabulary complexity (0-1 scale)"""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0

        unique_words = set(words)
        vocabulary_ratio = len(unique_words) / len(words)

        # Normalize to 0-1 scale (typical range 0.1-0.5 for natural text)
        complexity = min(1.0, vocabulary_ratio * 2)

        return complexity

    def _detect_document_type(self, text: str) -> str:
        """Auto-detect document type using pattern matching"""
        text_lower = text.lower()

        # Document type patterns (same as in the UI)
        patterns = {
            'technical': [
                r'machine learning', r'artificial intelligence', r'algorithm', r'neural network',
                r'deep learning', r'supervised learning', r'unsupervised learning'
            ],
            'legal': [
                r'agreement', r'terms of service', r'contract', r'liability', r'dispute',
                r'arbitration', r'jurisdiction', r'party', r'shall', r'agree'
            ],
            'medical': [
                r'clinical', r'patient', r'treatment', r'study', r'trial', r'melanoma',
                r'immunotherapy', r'cancer', r'disease', r'therapy'
            ],
            'code': [
                r'api', r'endpoint', r'request', r'response', r'authentication', r'oauth',
                r'json', r'http', r'get', r'post', r'delete'
            ],
            'ecommerce': [
                r'product', r'warranty', r'specifications', r'features', r'price',
                r'customer support', r'shipping', r'review'
            ],
            'support': [
                r'customer', r'support', r'ticket', r'conversation', r'agent',
                r'email', r'password', r'reset', r'account'
            ]
        }

        scores = {}
        for doc_type, type_patterns in patterns.items():
            score = 0
            for pattern in type_patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            scores[doc_type] = score

        # Return the type with highest score, or 'general' if no clear match
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'general'

    def _score_strategy(self, strategy_name: str, strategy, analysis: Dict, doc_type: str) -> float:
        """Score how well a strategy fits the text characteristics"""

        score = 0.0

        # Base score from document type compatibility
        score += self.strategy_weights[strategy_name].get(doc_type, 0.5)

        # Length-based adjustments
        if analysis['length'] > 20000:  # Long documents
            if strategy_name == 'recursive':
                score += 0.2  # Hierarchical works well for long docs
            elif strategy_name == 'fixed_size':
                score -= 0.1  # Fixed size may create too many chunks

        elif analysis['length'] < 1000:  # Short documents
            if strategy_name == 'fixed_size':
                score += 0.1  # Simple approach works for short docs

        # Structure-based scoring
        if analysis['has_headers'] and analysis['structural_complexity'] > 0.5:
            if strategy_name == 'recursive':
                score += 0.3  # Preserve document structure
            elif strategy_name == 'semantic':
                score += 0.2

        # Content complexity scoring
        if analysis['semantic_density'] > 0.7:  # Dense technical content
            if strategy_name == 'semantic':
                score += 0.2
            elif strategy_name == 'hybrid':
                score += 0.15

        if analysis['vocabulary_complexity'] > 0.6:  # Complex vocabulary
            if strategy_name == 'semantic':
                score += 0.15

        # Sentence structure scoring
        if analysis['avg_sentence_length'] > 50:  # Long, complex sentences
            if strategy_name == 'semantic':
                score += 0.2
            elif strategy_name == 'recursive':
                score += 0.1

        # Code content scoring
        if analysis['code_blocks'] > 0:
            if strategy_name == 'fixed_size':
                score += 0.2  # Code benefits from consistent chunking

        return min(1.0, max(0.0, score))

    def _optimize_strategy_parameters(self, strategy, analysis: Dict):
        """Dynamically adjust strategy parameters based on content"""

        if isinstance(strategy, FixedSizeChunking):
            # Adjust chunk size based on content complexity
            base_size = 400

            if analysis['semantic_density'] > 0.8:
                chunk_size = base_size * 0.8  # Smaller chunks for dense content
            elif analysis['avg_sentence_length'] > 40:
                chunk_size = base_size * 1.2  # Larger chunks for complex sentences
            elif analysis['length'] < 2000:  # Short documents
                chunk_size = base_size * 0.7  # Smaller chunks for short docs
            else:
                chunk_size = base_size

            strategy.chunk_size = int(chunk_size)

        elif isinstance(strategy, SlidingWindowChunking):
            base_window = 400
            base_step = 200

            if analysis['semantic_density'] > 0.7:
                strategy.window_size = int(base_window * 0.9)
                strategy.step_size = int(base_step * 0.8)
            else:
                strategy.window_size = base_window
                strategy.step_size = base_step

        elif isinstance(strategy, SemanticChunking):
            # Adjust max chunk size based on document structure
            if analysis['has_headers'] or analysis['structural_complexity'] > 0.6:
                strategy.max_chunk_size = 600  # Allow larger chunks with structure
            elif analysis['avg_sentence_length'] > 35:
                strategy.max_chunk_size = 500
            else:
                strategy.max_chunk_size = 400

        elif isinstance(strategy, RecursiveChunking):
            # Adjust based on document length and structure
            if analysis['length'] > 15000:
                strategy.max_chunk_size = 1000
            elif analysis['structural_complexity'] > 0.7:
                strategy.max_chunk_size = 800
            else:
                strategy.max_chunk_size = 600

        return strategy

    def _generate_reasoning(self, strategy_name: str, analysis: Dict, doc_type: str) -> str:
        """Generate human-readable reasoning for the recommendation"""

        reasons = {
            'fixed_size': "Provides consistent, predictable chunk sizes ideal for structured content and code.",
            'sliding_window': "Creates overlapping chunks that maintain context across boundaries.",
            'semantic': "Preserves semantic meaning by respecting sentence and paragraph boundaries.",
            'recursive': "Maintains document structure and hierarchy for organized content.",
            'hybrid': "Combines multiple approaches for balanced performance across different content types."
        }

        base_reason = reasons.get(strategy_name, "General-purpose chunking strategy.")

        # Add specific insights based on analysis
        insights = []

        if analysis['length'] > 20000 and strategy_name == 'recursive':
            insights.append("well-suited for long documents")

        if analysis['has_headers'] and strategy_name == 'recursive':
            insights.append("preserves document structure")

        if analysis['semantic_density'] > 0.7 and strategy_name == 'semantic':
            insights.append("maintains semantic coherence")

        if analysis['code_blocks'] > 0 and strategy_name == 'fixed_size':
            insights.append("provides consistent chunking for code")

        if doc_type != 'general':
            insights.append(f"optimized for {doc_type} content")

        if insights:
            return f"{base_reason} This strategy is {', '.join(insights)}."
        else:
            return base_reason


class LearningRecommender(DynamicChunkingRecommender):
    """Enhanced recommender with learning capabilities"""

    def __init__(self):
        super().__init__()
        self.performance_history = defaultdict(list)

    def recommend_with_feedback(self, text: str, user_feedback: Dict = None, doc_type: str = None):
        """Recommend strategy and incorporate user feedback"""

        recommendation = self.recommend_strategy(text, doc_type)

        if user_feedback:
            self._update_performance_history(recommendation, user_feedback)

        return recommendation

    def _update_performance_history(self, recommendation: Dict, feedback: Dict):
        """Update recommendation performance based on user feedback"""

        strategy_name = recommendation['strategy_name']

        performance_data = {
            'doc_type': feedback.get('doc_type', recommendation.get('doc_type')),
            'user_rating': feedback.get('rating', 0.5),
            'efficiency': feedback.get('efficiency', 0.5),
            'timestamp': datetime.now(),
            'text_length': feedback.get('text_length', 0),
            'strategy_score': recommendation['confidence']
        }

        self.performance_history[strategy_name].append(performance_data)

        # Update strategy weights based on performance (simple learning)
        self._update_weights_from_history()

    def _update_weights_from_history(self):
        """Update strategy weights based on historical performance"""

        # Simple learning: boost weights for strategies with good user ratings
        for strategy_name in self.strategy_weights:
            if strategy_name in self.performance_history:
                history = self.performance_history[strategy_name][-10:]  # Last 10 ratings

                if history:
                    avg_rating = np.mean([h['user_rating'] for h in history])
                    avg_efficiency = np.mean([h['efficiency'] for h in history])

                    # Boost weights slightly for well-performing strategies
                    boost = (avg_rating + avg_efficiency - 1.0) * 0.1

                    for doc_type in self.strategy_weights[strategy_name]:
                        self.strategy_weights[strategy_name][doc_type] = min(
                            1.0, self.strategy_weights[strategy_name][doc_type] + boost
                        )