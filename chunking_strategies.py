"""
RAG Chunking Strategy Implementation
Demonstrates various chunking techniques for Retrieval-Augmented Generation
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    chunk_id: int
    source: str
    start_char: int
    end_char: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
    @abstractmethod
    def chunk(self, text: str, source: str = "default") -> List[Chunk]:
        """Split text into chunks"""
        pass

class FixedSizeChunking(ChunkingStrategy):
    """
    Fixed-size chunking strategy
    Splits text into chunks of fixed token/character count
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 0):
        """
        Args:
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, source: str = "default") -> List[Chunk]:
        """Split text into fixed-size chunks"""
        chunks = []
        chunk_id = 0
        
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source=source,
                start_char=start,
                end_char=end
            ))
            
            chunk_id += 1
            start = end - self.overlap
        
        return chunks

class SlidingWindowChunking(ChunkingStrategy):
    """
    Sliding window chunking strategy
    Creates overlapping chunks using a sliding window approach
    """
    
    def __init__(self, window_size: int = 500, step_size: int = 250):
        """
        Args:
            window_size: Size of each chunk
            step_size: Step size for sliding window
        """
        self.window_size = window_size
        self.step_size = step_size
    
    def chunk(self, text: str, source: str = "default") -> List[Chunk]:
        """Create overlapping chunks using sliding window"""
        chunks = []
        chunk_id = 0
        
        for start in range(0, len(text), self.step_size):
            end = min(start + self.window_size, len(text))
            
            if end - start < self.window_size // 2:  # Skip if too small
                break
            
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source=source,
                start_char=start,
                end_char=end
            ))
            
            chunk_id += 1
        
        return chunks

class SemanticChunking(ChunkingStrategy):
    """
    Semantic chunking strategy
    Splits text based on sentences and paragraphs while respecting boundaries
    """
    
    def __init__(self, max_chunk_size: int = 500, min_chunk_size: int = 100):
        """
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Split by common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(r'\n\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk(self, text: str, source: str = "default") -> List[Chunk]:
        """Split text into semantic chunks"""
        chunks = []
        chunk_id = 0
        
        paragraphs = self._split_into_paragraphs(text)
        current_chunk = ""
        start_char = 0
        current_start = 0
        
        for para in paragraphs:
            sentences = self._split_into_sentences(para)
            
            for sentence in sentences:
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(test_chunk) > self.max_chunk_size:
                    # Save current chunk if it's large enough
                    if current_chunk and len(current_chunk) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            text=current_chunk,
                            chunk_id=chunk_id,
                            source=source,
                            start_char=current_start,
                            end_char=current_start + len(current_chunk)
                        ))
                        chunk_id += 1
                    current_chunk = sentence
                    current_start = start_char + len(current_chunk)
                else:
                    current_chunk = test_chunk
        
        # Add remaining chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(Chunk(
                text=current_chunk,
                chunk_id=chunk_id,
                source=source,
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            ))
        
        return chunks

class RecursiveChunking(ChunkingStrategy):
    """
    Recursive/Hierarchical chunking strategy
    Splits text based on document structure (headers, sections, etc.)
    """
    
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Args:
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _extract_sections(self, text: str) -> List[Tuple[str, int]]:
        """Extract sections based on markdown headers"""
        lines = text.split('\n')
        sections = []
        current_section = ""
        
        for i, line in enumerate(lines):
            if re.match(r'^#+\s', line):  # Header line
                if current_section:
                    sections.append((current_section, i))
                current_section = line
            else:
                current_section += '\n' + line if current_section else line
        
        if current_section:
            sections.append((current_section, len(lines)))
        
        return sections
    
    def chunk(self, text: str, source: str = "default") -> List[Chunk]:
        """Split text hierarchically"""
        chunks = []
        chunk_id = 0
        
        sections = self._extract_sections(text)
        
        for section_text, _ in sections:
            if len(section_text) <= self.max_chunk_size:
                chunks.append(Chunk(
                    text=section_text,
                    chunk_id=chunk_id,
                    source=source,
                    start_char=0,
                    end_char=len(section_text),
                    metadata={"type": "section"}
                ))
                chunk_id += 1
            else:
                # Further split large sections
                sub_chunks = self._split_large_section(section_text, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
        
        return chunks
    
    def _split_large_section(self, text: str, start_id: int) -> List[Chunk]:
        """Split large sections into smaller chunks"""
        chunks = []
        chunk_id = start_id
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(Chunk(
                        text=current_chunk,
                        chunk_id=chunk_id,
                        source="default",
                        start_char=0,
                        end_char=len(current_chunk),
                        metadata={"type": "subsection"}
                    ))
                    chunk_id += 1
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(Chunk(
                text=current_chunk,
                chunk_id=chunk_id,
                source="default",
                start_char=0,
                end_char=len(current_chunk),
                metadata={"type": "subsection"}
            ))
        
        return chunks

class HybridChunking(ChunkingStrategy):
    """
    Hybrid chunking strategy
    Combines multiple chunking approaches for optimal results
    """
    
    def __init__(self, semantic_weight: float = 0.6, window_weight: float = 0.4):
        """
        Args:
            semantic_weight: Weight for semantic chunking
            window_weight: Weight for sliding window chunking
        """
        self.semantic_weight = semantic_weight
        self.window_weight = window_weight
        self.semantic_chunker = SemanticChunking()
        self.window_chunker = SlidingWindowChunking()
    
    def chunk(self, text: str, source: str = "default") -> List[Chunk]:
        """Combine semantic and window-based chunking"""
        semantic_chunks = self.semantic_chunker.chunk(text, source)
        
        # Enrich semantic chunks with context from sliding window
        enriched_chunks = []
        for chunk in semantic_chunks:
            chunk.metadata['strategy'] = 'hybrid'
            chunk.metadata['semantic_score'] = 0.6
            enriched_chunks.append(chunk)
        
        return enriched_chunks

class ChunkAnalyzer:
    """Analyzes and evaluates chunk quality"""
    
    @staticmethod
    def analyze_chunks(chunks: List[Chunk]) -> Dict:
        """Analyze chunk statistics"""
        if not chunks:
            return {}
        
        sizes = [len(chunk.text) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': np.mean(sizes),
            'min_chunk_size': np.min(sizes),
            'max_chunk_size': np.max(sizes),
            'std_dev': np.std(sizes),
            'total_text_size': sum(sizes)
        }
    
    @staticmethod
    def print_chunks(chunks: List[Chunk], max_preview: int = 100):
        """Print chunk preview"""
        for chunk in chunks:
            preview = chunk.text[:max_preview].replace('\n', ' ')
            print(f"Chunk {chunk.chunk_id}: {preview}...")
            print(f"  Size: {len(chunk.text)} chars | Source: {chunk.source}\n")