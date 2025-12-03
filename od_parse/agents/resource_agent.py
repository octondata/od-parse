"""
Intelligent Resource Agent - Manages system resources dynamically.

This agent:
- Monitors system resources (CPU, memory)
- Dynamically adjusts parallel workers
- Prevents memory overflow
- Optimizes resource allocation
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import psutil

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    cpu_count: int


class ResourceAgent:
    """
    Intelligent resource agent that monitors and manages system resources.

    Features:
    - Real-time resource monitoring
    - Dynamic worker allocation
    - Memory overflow prevention
    - Adaptive quality adjustment
    """

    def __init__(
        self,
        max_cpu_percent: float = 80.0,
        max_memory_percent: float = 75.0,
        min_workers: int = 1,
        max_workers: int = 8,
    ):
        """
        Initialize the resource agent.

        Args:
            max_cpu_percent: Maximum CPU usage percentage
            max_memory_percent: Maximum memory usage percentage
            min_workers: Minimum parallel workers
            max_workers: Maximum parallel workers
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.min_workers = min_workers
        self.max_workers = max_workers

        # Get system info
        self.cpu_count = psutil.cpu_count()
        self.total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)

        logger.info(
            f"ResourceAgent initialized: CPU cores={self.cpu_count}, "
            f"Total memory={self.total_memory_mb:.0f}MB"
        )

    def get_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        memory = psutil.virtual_memory()

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_available_mb=memory.available / (1024 * 1024),
            cpu_count=self.cpu_count,
        )

    def recommend_workers(
        self, requested_workers: int, estimated_memory_per_worker_mb: float = 200.0
    ) -> int:
        """
        Recommend optimal number of workers based on current resources.

        Args:
            requested_workers: Number of workers requested
            estimated_memory_per_worker_mb: Estimated memory per worker

        Returns:
            Recommended number of workers
        """
        snapshot = self.get_snapshot()

        # CPU-based limit
        cpu_workers = max(1, int(self.cpu_count * (1 - snapshot.cpu_percent / 100)))

        # Memory-based limit
        available_memory = snapshot.memory_available_mb * 0.8  # Safety margin
        memory_workers = max(1, int(available_memory / estimated_memory_per_worker_mb))

        # Take minimum of all constraints
        recommended = min(
            requested_workers, cpu_workers, memory_workers, self.max_workers
        )
        recommended = max(recommended, self.min_workers)

        if recommended < requested_workers:
            logger.warning(
                f"Reducing workers from {requested_workers} to {recommended} "
                f"due to resource constraints (CPU={snapshot.cpu_percent:.1f}%, "
                f"Memory={snapshot.memory_percent:.1f}%)"
            )

        return recommended

    def check_memory_available(self, required_mb: float) -> bool:
        """
        Check if enough memory is available.

        Args:
            required_mb: Required memory in MB

        Returns:
            True if enough memory is available
        """
        snapshot = self.get_snapshot()
        available = snapshot.memory_available_mb * 0.8  # Safety margin

        if available < required_mb:
            logger.warning(
                f"Insufficient memory: required={required_mb:.0f}MB, "
                f"available={available:.0f}MB"
            )
            return False

        return True

    def recommend_image_quality(self, requested_dpi: int, page_count: int) -> int:
        """
        Recommend image DPI based on available memory.

        Args:
            requested_dpi: Requested DPI
            page_count: Number of pages

        Returns:
            Recommended DPI
        """
        snapshot = self.get_snapshot()

        # Estimate memory needed for requested DPI
        memory_per_page = (requested_dpi / 200) ** 2 * 50  # MB
        total_memory_needed = memory_per_page * page_count

        # Check if we have enough memory
        available = snapshot.memory_available_mb * 0.6  # Conservative

        if total_memory_needed <= available:
            return requested_dpi

        # Calculate reduced DPI
        ratio = (available / total_memory_needed) ** 0.5
        recommended_dpi = int(requested_dpi * ratio)
        recommended_dpi = max(150, min(recommended_dpi, requested_dpi))

        logger.warning(
            f"Reducing DPI from {requested_dpi} to {recommended_dpi} "
            f"due to memory constraints"
        )

        return recommended_dpi

    def should_throttle(self) -> bool:
        """
        Check if processing should be throttled due to resource constraints.

        Returns:
            True if should throttle
        """
        snapshot = self.get_snapshot()

        if snapshot.cpu_percent > self.max_cpu_percent:
            logger.warning(f"CPU usage high: {snapshot.cpu_percent:.1f}%")
            return True

        if snapshot.memory_percent > self.max_memory_percent:
            logger.warning(f"Memory usage high: {snapshot.memory_percent:.1f}%")
            return True

        return False

    def wait_for_resources(self, timeout: float = 30.0):
        """
        Wait for resources to become available.

        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()

        while self.should_throttle():
            if time.time() - start_time > timeout:
                logger.warning("Timeout waiting for resources")
                break

            logger.info("Waiting for resources to become available...")
            time.sleep(1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get resource statistics."""
        snapshot = self.get_snapshot()

        return {
            "cpu_percent": snapshot.cpu_percent,
            "cpu_count": snapshot.cpu_count,
            "memory_percent": snapshot.memory_percent,
            "memory_available_mb": snapshot.memory_available_mb,
            "memory_total_mb": self.total_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_percent": self.max_memory_percent,
        }
