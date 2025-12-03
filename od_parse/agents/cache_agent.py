"""
Intelligent Cache Agent - Manages caching with smart eviction and prioritization.

This agent:
- Decides what to cache based on access patterns
- Manages cache eviction intelligently
- Predicts cache hits
- Optimizes cache storage
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    file_hash: str
    result: Dict[str, Any]
    created_at: float
    last_accessed: float
    access_count: int
    file_size: int
    cache_size: int
    priority_score: float


class CacheAgent:
    """
    Intelligent cache agent that manages document parsing cache.

    Features:
    - LRU + LFU hybrid eviction
    - Priority-based caching
    - Automatic cache warming
    - Access pattern learning
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_mb: int = 512,
        max_disk_mb: int = 2048,
        enable_disk_cache: bool = True,
    ):
        """
        Initialize the cache agent.

        Args:
            cache_dir: Directory for disk cache
            max_memory_mb: Maximum memory cache size in MB
            max_disk_mb: Maximum disk cache size in MB
            enable_disk_cache: Whether to use disk cache
        """
        self.cache_dir = cache_dir or Path.home() / ".od_parse" / "cache"
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self.enable_disk_cache = enable_disk_cache

        # In-memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.current_memory_size = 0

        # Access patterns for learning
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)

        # Create cache directory
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Disk cache enabled at: {self.cache_dir}")

        logger.info(
            f"CacheAgent initialized: memory={max_memory_mb}MB, "
            f"disk={max_disk_mb}MB"
        )

    def get(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a document.

        Args:
            file_hash: Hash of the document

        Returns:
            Cached result or None if not found
        """
        # Check memory cache first
        if file_hash in self.memory_cache:
            entry = self.memory_cache[file_hash]
            entry.last_accessed = time.time()
            entry.access_count += 1

            # Record access pattern
            self.access_patterns[file_hash].append(time.time())

            logger.info(
                f"Cache HIT (memory): {file_hash[:8]}... "
                f"(accessed {entry.access_count} times)"
            )
            return entry.result

        # Check disk cache
        if self.enable_disk_cache:
            disk_result = self._load_from_disk(file_hash)
            if disk_result is not None:
                # Promote to memory cache
                self._add_to_memory(file_hash, disk_result)
                logger.info(f"Cache HIT (disk): {file_hash[:8]}...")
                return disk_result

        logger.info(f"Cache MISS: {file_hash[:8]}...")
        return None

    def put(
        self,
        file_hash: str,
        result: Dict[str, Any],
        file_size: int,
        priority: float = 1.0,
    ):
        """
        Store result in cache with intelligent prioritization.

        Args:
            file_hash: Hash of the document
            result: Parsing result to cache
            file_size: Size of original file
            priority: Priority score (higher = more important)
        """
        # Calculate cache size
        cache_size = len(pickle.dumps(result))

        # Decide whether to cache based on size and priority
        if not self._should_cache(cache_size, priority):
            logger.info(
                f"Skipping cache (low priority or too large): {file_hash[:8]}..."
            )
            return

        # Create cache entry
        entry = CacheEntry(
            file_hash=file_hash,
            result=result,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            file_size=file_size,
            cache_size=cache_size,
            priority_score=priority,
        )

        # Add to memory cache
        self._add_to_memory_with_eviction(file_hash, entry)

        # Add to disk cache
        if self.enable_disk_cache:
            self._save_to_disk(file_hash, result)

        logger.info(
            f"Cached result: {file_hash[:8]}... "
            f"(size={cache_size/1024:.1f}KB, priority={priority:.2f})"
        )

    def _should_cache(self, cache_size: int, priority: float) -> bool:
        """Decide whether to cache based on size and priority."""
        # Don't cache if too large
        if cache_size > self.max_memory_bytes * 0.5:
            return False

        # Always cache high priority items
        if priority > 2.0:
            return True

        # Cache based on size/priority ratio
        size_mb = cache_size / (1024 * 1024)
        return priority > size_mb * 0.1

    def _add_to_memory(self, file_hash: str, result: Dict[str, Any]):
        """Add result to memory cache (for disk cache promotion)."""
        cache_size = len(pickle.dumps(result))

        entry = CacheEntry(
            file_hash=file_hash,
            result=result,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            file_size=0,
            cache_size=cache_size,
            priority_score=1.0,
        )

        self._add_to_memory_with_eviction(file_hash, entry)

    def _add_to_memory_with_eviction(self, file_hash: str, entry: CacheEntry):
        """Add entry to memory cache with intelligent eviction."""
        # Check if already in cache
        if file_hash in self.memory_cache:
            old_entry = self.memory_cache[file_hash]
            self.current_memory_size -= old_entry.cache_size

        # Evict if necessary
        while (
            self.current_memory_size + entry.cache_size > self.max_memory_bytes
            and self.memory_cache
        ):
            self._evict_one()

        # Add to cache
        self.memory_cache[file_hash] = entry
        self.current_memory_size += entry.cache_size

    def _evict_one(self):
        """Evict one entry using intelligent scoring."""
        if not self.memory_cache:
            return

        # Calculate eviction scores (lower = evict first)
        scores = {}
        current_time = time.time()

        for file_hash, entry in self.memory_cache.items():
            # Factors:
            # - Recency (last accessed)
            # - Frequency (access count)
            # - Priority
            # - Size (prefer evicting large items)

            recency_score = 1.0 / (current_time - entry.last_accessed + 1)
            frequency_score = entry.access_count
            priority_score = entry.priority_score
            size_penalty = entry.cache_size / (1024 * 1024)  # MB

            # Combined score (higher = keep)
            score = (
                recency_score * 0.3
                + frequency_score * 0.3
                + priority_score * 0.3
                - size_penalty * 0.1
            )

            scores[file_hash] = score

        # Evict lowest score
        evict_hash = min(scores, key=scores.get)
        evict_entry = self.memory_cache[evict_hash]

        logger.info(
            f"Evicting from memory cache: {evict_hash[:8]}... "
            f"(score={scores[evict_hash]:.3f}, "
            f"size={evict_entry.cache_size/1024:.1f}KB)"
        )

        self.current_memory_size -= evict_entry.cache_size
        del self.memory_cache[evict_hash]

    def _save_to_disk(self, file_hash: str, result: Dict[str, Any]):
        """Save result to disk cache."""
        try:
            cache_file = self.cache_dir / f"{file_hash}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.error(f"Error saving to disk cache: {e}")

    def _load_from_disk(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Load result from disk cache."""
        try:
            cache_file = self.cache_dir / f"{file_hash}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading from disk cache: {e}")
        return None

    def predict_cache_hit(self, file_hash: str) -> float:
        """
        Predict probability of cache hit based on access patterns.

        Args:
            file_hash: Hash of the document

        Returns:
            Probability of cache hit (0.0 to 1.0)
        """
        if file_hash in self.memory_cache:
            return 1.0

        if file_hash in self.access_patterns:
            # Analyze access pattern
            accesses = self.access_patterns[file_hash]
            if len(accesses) >= 2:
                # Calculate average time between accesses
                intervals = [
                    accesses[i + 1] - accesses[i] for i in range(len(accesses) - 1)
                ]
                avg_interval = sum(intervals) / len(intervals)

                # Predict based on time since last access
                time_since_last = time.time() - accesses[-1]
                probability = max(0.0, 1.0 - (time_since_last / avg_interval))
                return probability

        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_accesses = sum(e.access_count for e in self.memory_cache.values())

        return {
            "memory_entries": len(self.memory_cache),
            "memory_size_mb": self.current_memory_size / (1024 * 1024),
            "memory_utilization": self.current_memory_size / self.max_memory_bytes,
            "total_accesses": total_accesses,
            "avg_accesses_per_entry": (
                total_accesses / len(self.memory_cache) if self.memory_cache else 0
            ),
            "tracked_patterns": len(self.access_patterns),
        }

    def clear(self):
        """Clear all caches."""
        self.memory_cache.clear()
        self.current_memory_size = 0

        if self.enable_disk_cache:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Error deleting cache file: {e}")

        logger.info("Cache cleared")

    def warm_cache(self, file_hashes: List[str]):
        """
        Warm cache by preloading frequently accessed documents.

        Args:
            file_hashes: List of file hashes to preload
        """
        logger.info(f"Warming cache with {len(file_hashes)} documents...")

        for file_hash in file_hashes:
            if file_hash not in self.memory_cache:
                result = self._load_from_disk(file_hash)
                if result:
                    self._add_to_memory(file_hash, result)

        logger.info("Cache warming complete")
