"""
Intelligent Parsing Agent - Makes smart decisions about document processing.

This agent analyzes documents and decides:
- Whether to use cache
- How many parallel workers to use
- What quality settings to apply
- Which extraction methods to prioritize
"""

import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProcessingStrategy(Enum):
    """Processing strategies based on document analysis."""

    FAST = "fast"  # Speed-optimized, lower quality
    BALANCED = "balanced"  # Balance speed and quality
    ACCURATE = "accurate"  # Quality-optimized, slower
    TABLES_ONLY = "tables_only"  # Focus on table extraction
    TEXT_ONLY = "text_only"  # Text extraction only
    ADAPTIVE = "adaptive"  # Agent decides dynamically


@dataclass
class DocumentProfile:
    """Profile of a document for intelligent processing."""

    file_path: Path
    file_size: int
    page_count: int
    file_hash: str
    estimated_complexity: str  # 'simple', 'medium', 'complex'
    has_images: bool
    has_tables: bool
    has_forms: bool
    language: str
    is_scanned: bool


@dataclass
class ProcessingPlan:
    """Intelligent processing plan created by the agent."""

    strategy: ProcessingStrategy
    use_cache: bool
    parallel_workers: int
    max_pages_for_images: Optional[int]
    image_dpi: int
    extract_text: bool
    extract_images: bool
    extract_tables: bool
    extract_forms: bool
    use_llm: bool
    estimated_time: float
    estimated_memory_mb: float
    reasoning: str  # Agent's reasoning for this plan


class ParsingAgent:
    """
    Intelligent agent that analyzes documents and creates optimal processing plans.

    This agent uses heuristics and learned patterns to make smart decisions about
    how to process each document for optimal performance.
    """

    def __init__(
        self,
        strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE,
        max_workers: int = 4,
        max_memory_mb: int = 2048,
        enable_learning: bool = True,
    ):
        """
        Initialize the parsing agent.

        Args:
            strategy: Default processing strategy
            max_workers: Maximum parallel workers available
            max_memory_mb: Maximum memory budget in MB
            enable_learning: Whether to learn from processing history
        """
        self.strategy = strategy
        self.max_workers = max_workers
        self.max_memory_mb = max_memory_mb
        self.enable_learning = enable_learning

        # Learning: Track processing history
        self.processing_history: List[Dict[str, Any]] = []

        logger.info(
            f"ParsingAgent initialized with strategy={strategy.value}, "
            f"max_workers={max_workers}, max_memory={max_memory_mb}MB"
        )

    def analyze_document(self, file_path: Path) -> DocumentProfile:
        """
        Analyze a document to understand its characteristics.

        Args:
            file_path: Path to the PDF file

        Returns:
            DocumentProfile with document characteristics
        """
        logger.info(f"Analyzing document: {file_path}")

        # Get file size
        file_size = file_path.stat().st_size

        # Calculate file hash for caching
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Get page count
        try:
            from pdfminer.pdfpage import PDFPage

            with open(file_path, "rb") as f:
                page_count = len(list(PDFPage.get_pages(f)))
        except Exception as e:
            logger.warning(f"Could not determine page count: {e}")
            page_count = 1

        # Quick analysis: Check first page for content types
        has_images = self._quick_check_images(file_path)
        has_tables = self._quick_check_tables(file_path)
        has_forms = self._quick_check_forms(file_path)
        is_scanned = self._quick_check_scanned(file_path)

        # Estimate complexity
        complexity = self._estimate_complexity(
            file_size, page_count, has_images, has_tables, has_forms
        )

        profile = DocumentProfile(
            file_path=file_path,
            file_size=file_size,
            page_count=page_count,
            file_hash=file_hash,
            estimated_complexity=complexity,
            has_images=has_images,
            has_tables=has_tables,
            has_forms=has_forms,
            language="en",  # TODO: Add language detection
            is_scanned=is_scanned,
        )

        logger.info(
            f"Document profile: {page_count} pages, {complexity} complexity, "
            f"images={has_images}, tables={has_tables}, scanned={is_scanned}"
        )

        return profile

    def create_processing_plan(
        self,
        profile: DocumentProfile,
        user_strategy: Optional[ProcessingStrategy] = None,
    ) -> ProcessingPlan:
        """
        Create an intelligent processing plan based on document profile.

        Args:
            profile: Document profile from analysis
            user_strategy: Optional user-specified strategy (overrides agent)

        Returns:
            ProcessingPlan with optimal settings
        """
        strategy = user_strategy or self.strategy

        # Agent decision-making logic
        if strategy == ProcessingStrategy.ADAPTIVE:
            plan = self._create_adaptive_plan(profile)
        elif strategy == ProcessingStrategy.FAST:
            plan = self._create_fast_plan(profile)
        elif strategy == ProcessingStrategy.ACCURATE:
            plan = self._create_accurate_plan(profile)
        elif strategy == ProcessingStrategy.TABLES_ONLY:
            plan = self._create_tables_only_plan(profile)
        elif strategy == ProcessingStrategy.TEXT_ONLY:
            plan = self._create_text_only_plan(profile)
        else:
            plan = self._create_balanced_plan(profile)

        logger.info(
            f"Processing plan created: {plan.strategy.value}, "
            f"workers={plan.parallel_workers}, cache={plan.use_cache}, "
            f"est_time={plan.estimated_time:.1f}s, est_memory={plan.estimated_memory_mb:.0f}MB"
        )
        logger.info(f"Agent reasoning: {plan.reasoning}")

        return plan

    def _create_adaptive_plan(self, profile: DocumentProfile) -> ProcessingPlan:
        """Create an adaptive plan based on document characteristics."""

        # Decision logic based on document profile
        reasoning_parts = []

        # Decide on caching
        use_cache = True
        reasoning_parts.append("Caching enabled for potential reuse")

        # Decide on parallel workers based on complexity and page count
        if profile.page_count <= 5:
            parallel_workers = 2
            reasoning_parts.append("Few pages: using 2 workers")
        elif profile.page_count <= 20:
            parallel_workers = 4
            reasoning_parts.append("Medium document: using 4 workers")
        else:
            parallel_workers = min(self.max_workers, 6)
            reasoning_parts.append(f"Large document: using {parallel_workers} workers")

        # Decide on image processing
        if profile.is_scanned:
            max_pages_for_images = profile.page_count  # Process all pages
            image_dpi = 300  # High quality for OCR
            reasoning_parts.append("Scanned document: high-quality image processing")
        elif profile.has_images:
            max_pages_for_images = min(profile.page_count, 10)
            image_dpi = 200  # Medium quality
            reasoning_parts.append(
                "Has images: medium-quality processing for first 10 pages"
            )
        else:
            max_pages_for_images = 3
            image_dpi = 150  # Low quality
            reasoning_parts.append("No images detected: minimal image processing")

        # Decide what to extract
        extract_text = True
        extract_images = profile.has_images or profile.is_scanned
        extract_tables = profile.has_tables
        extract_forms = profile.has_forms

        # Decide on LLM usage
        use_llm = profile.estimated_complexity in ["medium", "complex"]
        if use_llm:
            reasoning_parts.append("Complex document: LLM enhancement enabled")

        # Estimate resources
        estimated_time = self._estimate_time(profile, parallel_workers)
        estimated_memory_mb = self._estimate_memory(
            profile, image_dpi, max_pages_for_images
        )

        return ProcessingPlan(
            strategy=ProcessingStrategy.ADAPTIVE,
            use_cache=use_cache,
            parallel_workers=parallel_workers,
            max_pages_for_images=max_pages_for_images,
            image_dpi=image_dpi,
            extract_text=extract_text,
            extract_images=extract_images,
            extract_tables=extract_tables,
            extract_forms=extract_forms,
            use_llm=use_llm,
            estimated_time=estimated_time,
            estimated_memory_mb=estimated_memory_mb,
            reasoning="; ".join(reasoning_parts),
        )

    def _create_fast_plan(self, profile: DocumentProfile) -> ProcessingPlan:
        """Create a speed-optimized plan."""
        return ProcessingPlan(
            strategy=ProcessingStrategy.FAST,
            use_cache=True,
            parallel_workers=self.max_workers,
            max_pages_for_images=3,
            image_dpi=150,
            extract_text=True,
            extract_images=False,
            extract_tables=profile.has_tables,
            extract_forms=False,
            use_llm=False,
            estimated_time=self._estimate_time(profile, self.max_workers) * 0.5,
            estimated_memory_mb=self._estimate_memory(profile, 150, 3),
            reasoning="Fast mode: minimal processing, maximum speed",
        )

    def _create_accurate_plan(self, profile: DocumentProfile) -> ProcessingPlan:
        """Create a quality-optimized plan."""
        return ProcessingPlan(
            strategy=ProcessingStrategy.ACCURATE,
            use_cache=True,
            parallel_workers=2,  # Fewer workers for stability
            max_pages_for_images=profile.page_count,
            image_dpi=300,
            extract_text=True,
            extract_images=True,
            extract_tables=True,
            extract_forms=True,
            use_llm=True,
            estimated_time=self._estimate_time(profile, 2) * 2.0,
            estimated_memory_mb=self._estimate_memory(profile, 300, profile.page_count),
            reasoning="Accurate mode: comprehensive extraction, maximum quality",
        )

    def _create_balanced_plan(self, profile: DocumentProfile) -> ProcessingPlan:
        """Create a balanced plan."""
        return ProcessingPlan(
            strategy=ProcessingStrategy.BALANCED,
            use_cache=True,
            parallel_workers=4,
            max_pages_for_images=min(profile.page_count, 10),
            image_dpi=200,
            extract_text=True,
            extract_images=profile.has_images,
            extract_tables=profile.has_tables,
            extract_forms=profile.has_forms,
            use_llm=profile.estimated_complexity != "simple",
            estimated_time=self._estimate_time(profile, 4),
            estimated_memory_mb=self._estimate_memory(profile, 200, 10),
            reasoning="Balanced mode: good quality with reasonable speed",
        )

    def _create_tables_only_plan(self, profile: DocumentProfile) -> ProcessingPlan:
        """Create a tables-only plan."""
        return ProcessingPlan(
            strategy=ProcessingStrategy.TABLES_ONLY,
            use_cache=True,
            parallel_workers=2,
            max_pages_for_images=0,
            image_dpi=150,
            extract_text=False,
            extract_images=False,
            extract_tables=True,
            extract_forms=False,
            use_llm=False,
            estimated_time=self._estimate_time(profile, 2) * 0.3,
            estimated_memory_mb=200,
            reasoning="Tables-only mode: focused table extraction",
        )

    def _create_text_only_plan(self, profile: DocumentProfile) -> ProcessingPlan:
        """Create a text-only plan."""
        return ProcessingPlan(
            strategy=ProcessingStrategy.TEXT_ONLY,
            use_cache=True,
            parallel_workers=1,
            max_pages_for_images=0,
            image_dpi=150,
            extract_text=True,
            extract_images=False,
            extract_tables=False,
            extract_forms=False,
            use_llm=False,
            estimated_time=self._estimate_time(profile, 1) * 0.2,
            estimated_memory_mb=100,
            reasoning="Text-only mode: fast text extraction",
        )

    # Helper methods for quick analysis
    def _quick_check_images(self, file_path: Path) -> bool:
        """Quick check if document has images."""
        # TODO: Implement quick image detection
        return True  # Conservative assumption

    def _quick_check_tables(self, file_path: Path) -> bool:
        """Quick check if document has tables."""
        # TODO: Implement quick table detection
        return True  # Conservative assumption

    def _quick_check_forms(self, file_path: Path) -> bool:
        """Quick check if document has forms."""
        # TODO: Implement quick form detection
        return False

    def _quick_check_scanned(self, file_path: Path) -> bool:
        """Quick check if document is scanned."""
        # TODO: Implement scanned document detection
        return False

    def _estimate_complexity(
        self,
        file_size: int,
        page_count: int,
        has_images: bool,
        has_tables: bool,
        has_forms: bool,
    ) -> str:
        """Estimate document complexity."""
        complexity_score = 0

        # Size factor
        if file_size > 10 * 1024 * 1024:  # > 10MB
            complexity_score += 2
        elif file_size > 1 * 1024 * 1024:  # > 1MB
            complexity_score += 1

        # Page count factor
        if page_count > 50:
            complexity_score += 2
        elif page_count > 10:
            complexity_score += 1

        # Content factor
        if has_images:
            complexity_score += 1
        if has_tables:
            complexity_score += 1
        if has_forms:
            complexity_score += 1

        if complexity_score >= 5:
            return "complex"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "simple"

    def _estimate_time(self, profile: DocumentProfile, workers: int) -> float:
        """Estimate processing time in seconds."""
        base_time_per_page = 1.5  # seconds
        total_time = profile.page_count * base_time_per_page

        # Adjust for complexity
        if profile.estimated_complexity == "complex":
            total_time *= 2.0
        elif profile.estimated_complexity == "medium":
            total_time *= 1.5

        # Adjust for parallelization (not linear)
        speedup = min(workers, profile.page_count) * 0.7  # 70% efficiency
        total_time /= max(speedup, 1)

        return total_time

    def _estimate_memory(
        self, profile: DocumentProfile, dpi: int, max_pages: Optional[int]
    ) -> float:
        """Estimate memory usage in MB."""
        base_memory = 100  # Base overhead

        # Memory per page for images
        pages_to_process = min(profile.page_count, max_pages or profile.page_count)
        memory_per_page = (dpi / 200) ** 2 * 50  # MB, scales with DPI squared

        total_memory = base_memory + (pages_to_process * memory_per_page)

        return total_memory

    def record_processing_result(
        self,
        profile: DocumentProfile,
        plan: ProcessingPlan,
        actual_time: float,
        actual_memory_mb: float,
        success: bool,
    ):
        """
        Record processing result for learning.

        Args:
            profile: Document profile
            plan: Processing plan used
            actual_time: Actual processing time
            actual_memory_mb: Actual memory used
            success: Whether processing succeeded
        """
        if not self.enable_learning:
            return

        result = {
            "timestamp": time.time(),
            "file_hash": profile.file_hash,
            "page_count": profile.page_count,
            "complexity": profile.estimated_complexity,
            "strategy": plan.strategy.value,
            "workers": plan.parallel_workers,
            "estimated_time": plan.estimated_time,
            "actual_time": actual_time,
            "estimated_memory": plan.estimated_memory_mb,
            "actual_memory": actual_memory_mb,
            "success": success,
        }

        self.processing_history.append(result)

        # Keep only last 100 results
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-100:]

        logger.info(
            f"Recorded processing result: time={actual_time:.2f}s "
            f"(est={plan.estimated_time:.2f}s), success={success}"
        )
