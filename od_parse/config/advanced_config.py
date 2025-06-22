"""
Advanced Configuration System for od-parse

This module provides a flexible configuration system that allows users to 
enable/disable advanced features with optional dependencies.
"""

import os
import json
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureLevel(Enum):
    """Feature complexity levels"""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"


@dataclass
class FeatureConfig:
    """Configuration for individual features"""
    name: str
    enabled: bool = False
    level: FeatureLevel = FeatureLevel.BASIC
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    description: str = ""
    performance_impact: str = "low"  # low, medium, high
    memory_usage: str = "low"  # low, medium, high


class AdvancedConfig:
    """
    Advanced configuration manager for od-parse features.
    
    This class manages optional features and their dependencies,
    allowing users to enable only what they need.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.logger = get_logger(__name__)
        self.config_file = config_file
        self._features = self._initialize_features()
        self._dependency_cache = {}
        
        # Load configuration from file if provided
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def _initialize_features(self) -> Dict[str, FeatureConfig]:
        """Initialize all available features with their configurations."""
        return {
            # OCR Features
            "trocr": FeatureConfig(
                name="TrOCR",
                enabled=False,
                level=FeatureLevel.ADVANCED,
                dependencies=["torch", "transformers", "torchvision"],
                description="Transformer-based OCR for superior text recognition",
                performance_impact="medium",
                memory_usage="high"
            ),
            
            # Table Extraction Features
            "table_transformer": FeatureConfig(
                name="Table Transformer",
                enabled=False,
                level=FeatureLevel.ADVANCED,
                dependencies=["torch", "transformers", "detectron2"],
                description="Neural table detection and extraction",
                performance_impact="high",
                memory_usage="high"
            ),
            
            # Document Understanding Features
            "llava_next": FeatureConfig(
                name="LLaVA-NeXT",
                enabled=False,
                level=FeatureLevel.EXPERIMENTAL,
                dependencies=["torch", "transformers", "accelerate"],
                optional_dependencies=["bitsandbytes"],
                description="Advanced document understanding with vision-language models",
                performance_impact="high",
                memory_usage="very_high"
            ),
            
            # Quality Assessment
            "quality_assessment": FeatureConfig(
                name="Document Quality Assessment",
                enabled=False,
                level=FeatureLevel.BASIC,
                dependencies=["scikit-learn", "scipy"],
                description="Assess extraction quality and confidence scores",
                performance_impact="low",
                memory_usage="low"
            ),
            
            # Async Processing
            "async_processing": FeatureConfig(
                name="Async Processing",
                enabled=False,
                level=FeatureLevel.BASIC,
                dependencies=["aiofiles", "asyncio"],
                description="Asynchronous processing for large files",
                performance_impact="low",
                memory_usage="low"
            ),
            
            # Multi-language Support
            "multilingual": FeatureConfig(
                name="Multi-Language Support",
                enabled=False,
                level=FeatureLevel.ADVANCED,
                dependencies=["spacy", "langdetect", "polyglot"],
                optional_dependencies=["googletrans"],
                description="Comprehensive multilingual document processing",
                performance_impact="medium",
                memory_usage="medium"
            ),
            
            # Performance Features
            "gpu_acceleration": FeatureConfig(
                name="GPU Acceleration",
                enabled=False,
                level=FeatureLevel.ADVANCED,
                dependencies=["cupy-cuda11x"],
                description="GPU acceleration for image processing",
                performance_impact="high",
                memory_usage="high"
            ),
            
            # Caching
            "advanced_caching": FeatureConfig(
                name="Advanced Caching",
                enabled=False,
                level=FeatureLevel.BASIC,
                dependencies=["redis", "joblib"],
                description="Intelligent caching for improved performance",
                performance_impact="low",
                memory_usage="medium"
            )
        }
    
    def enable_feature(self, feature_name: str, check_dependencies: bool = True) -> bool:
        """
        Enable a specific feature.
        
        Args:
            feature_name: Name of the feature to enable
            check_dependencies: Whether to check if dependencies are available
            
        Returns:
            True if feature was enabled successfully, False otherwise
        """
        if feature_name not in self._features:
            self.logger.error(f"Unknown feature: {feature_name}")
            return False
        
        feature = self._features[feature_name]
        
        if check_dependencies:
            missing_deps = self._check_dependencies(feature)
            if missing_deps:
                self.logger.warning(
                    f"Cannot enable {feature_name}. Missing dependencies: {missing_deps}"
                )
                self.logger.info(
                    f"Install with: pip install od-parse[{feature_name}]"
                )
                return False
        
        feature.enabled = True
        self.logger.info(f"Enabled feature: {feature_name}")
        return True
    
    def disable_feature(self, feature_name: str) -> bool:
        """
        Disable a specific feature.
        
        Args:
            feature_name: Name of the feature to disable
            
        Returns:
            True if feature was disabled successfully, False otherwise
        """
        if feature_name not in self._features:
            self.logger.error(f"Unknown feature: {feature_name}")
            return False
        
        self._features[feature_name].enabled = False
        self.logger.info(f"Disabled feature: {feature_name}")
        return True
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return self._features.get(feature_name, FeatureConfig("")).enabled
    
    def is_feature_available(self, feature_name: str) -> bool:
        """Check if a feature is available (dependencies installed)."""
        if feature_name not in self._features:
            return False
        
        feature = self._features[feature_name]
        missing_deps = self._check_dependencies(feature)
        return len(missing_deps) == 0
    
    def _check_dependencies(self, feature: FeatureConfig) -> List[str]:
        """Check which dependencies are missing for a feature."""
        missing = []
        
        for dep in feature.dependencies:
            if not self._is_package_available(dep):
                missing.append(dep)
        
        return missing
    
    def _is_package_available(self, package_name: str) -> bool:
        """Check if a package is available for import."""
        if package_name in self._dependency_cache:
            return self._dependency_cache[package_name]
        
        try:
            __import__(package_name.replace("-", "_"))
            self._dependency_cache[package_name] = True
            return True
        except ImportError:
            self._dependency_cache[package_name] = False
            return False
    
    def get_enabled_features(self) -> List[str]:
        """Get list of enabled features."""
        return [name for name, config in self._features.items() if config.enabled]
    
    def get_available_features(self) -> List[str]:
        """Get list of available features (with dependencies installed)."""
        return [name for name in self._features.keys() if self.is_feature_available(name)]
    
    def get_feature_info(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a feature."""
        if feature_name not in self._features:
            return None
        
        feature = self._features[feature_name]
        return {
            "name": feature.name,
            "enabled": feature.enabled,
            "available": self.is_feature_available(feature_name),
            "level": feature.level.value,
            "description": feature.description,
            "dependencies": feature.dependencies,
            "optional_dependencies": feature.optional_dependencies,
            "performance_impact": feature.performance_impact,
            "memory_usage": feature.memory_usage
        }
    
    def list_all_features(self) -> Dict[str, Dict[str, Any]]:
        """List all features with their information."""
        return {name: self.get_feature_info(name) for name in self._features.keys()}
    
    def save_config(self, config_file: str) -> bool:
        """Save current configuration to file."""
        try:
            config_data = {
                "enabled_features": self.get_enabled_features(),
                "feature_configs": {
                    name: {
                        "enabled": config.enabled,
                        "level": config.level.value
                    }
                    for name, config in self._features.items()
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Configuration saved to {config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self, config_file: str) -> bool:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Enable features from config
            for feature_name in config_data.get("enabled_features", []):
                self.enable_feature(feature_name, check_dependencies=False)
            
            self.logger.info(f"Configuration loaded from {config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def enable_preset(self, preset_name: str) -> bool:
        """
        Enable a preset configuration.
        
        Available presets:
        - basic: Essential features only
        - advanced: All stable advanced features
        - experimental: All features including experimental ones
        - performance: Features focused on performance
        - quality: Features focused on extraction quality
        """
        presets = {
            "basic": ["quality_assessment", "async_processing"],
            "advanced": ["trocr", "table_transformer", "quality_assessment", "async_processing", "multilingual"],
            "experimental": ["trocr", "table_transformer", "llava_next", "quality_assessment", "async_processing", "multilingual"],
            "performance": ["gpu_acceleration", "advanced_caching", "async_processing"],
            "quality": ["trocr", "table_transformer", "quality_assessment", "multilingual"]
        }
        
        if preset_name not in presets:
            self.logger.error(f"Unknown preset: {preset_name}")
            return False
        
        success_count = 0
        for feature in presets[preset_name]:
            if self.enable_feature(feature):
                success_count += 1
        
        self.logger.info(f"Enabled {success_count}/{len(presets[preset_name])} features from preset '{preset_name}'")
        return success_count > 0


# Global configuration instance
_global_config = None


def get_config() -> AdvancedConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        # Check for config file in environment or default location
        config_file = os.getenv("OD_PARSE_CONFIG", None)
        _global_config = AdvancedConfig(config_file)
    return _global_config


def configure_features(**kwargs) -> AdvancedConfig:
    """
    Configure features programmatically.
    
    Args:
        **kwargs: Feature names as keys, boolean values to enable/disable
        
    Returns:
        The global configuration instance
    """
    config = get_config()
    
    for feature_name, enabled in kwargs.items():
        if enabled:
            config.enable_feature(feature_name)
        else:
            config.disable_feature(feature_name)
    
    return config
