"""
Document Quality Assessment Module

This module provides comprehensive quality assessment metrics for document
extraction results, helping users understand the reliability and accuracy
of the parsed content.
"""

import re
import math
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

from od_parse.config import get_advanced_config
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DocumentQualityAssessor:
    """
    Comprehensive document quality assessment engine.
    
    This class provides various metrics to assess the quality of document
    extraction results, including text quality, structure coherence,
    and confidence scores.
    """
    
    def __init__(self):
        """Initialize the quality assessor."""
        self.logger = get_logger(__name__)
        self._sklearn_available = False
        self._scipy_available = False
        
        # Check for optional dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for optional dependencies."""
        config = get_advanced_config()
        
        if not config.is_feature_enabled("quality_assessment"):
            self.logger.info("Quality assessment feature is disabled. Use config.enable_feature('quality_assessment') to enable.")
            return
        
        try:
            import sklearn
            self._sklearn_available = True
        except ImportError:
            self.logger.warning("scikit-learn not available. Some quality metrics will be limited.")
        
        try:
            import scipy
            self._scipy_available = True
        except ImportError:
            self.logger.warning("scipy not available. Some statistical metrics will be limited.")
    
    def assess_extraction_quality(
        self, 
        extraction_result: Dict[str, Any],
        original_image: Optional[Union[str, Path, Image.Image]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment of extraction results.
        
        Args:
            extraction_result: The result from document extraction
            original_image: Original document image (optional, for image-based metrics)
            
        Returns:
            Dictionary containing quality assessment metrics
        """
        try:
            assessment = {
                "overall_score": 0.0,
                "text_quality": self._assess_text_quality(extraction_result),
                "structure_quality": self._assess_structure_quality(extraction_result),
                "confidence_metrics": self._assess_confidence_metrics(extraction_result),
                "completeness": self._assess_completeness(extraction_result),
                "consistency": self._assess_consistency(extraction_result)
            }
            
            # Add image-based metrics if original image is provided
            if original_image is not None:
                assessment["image_quality"] = self._assess_image_quality(original_image)
            
            # Calculate overall score
            assessment["overall_score"] = self._calculate_overall_score(assessment)
            
            # Add recommendations
            assessment["recommendations"] = self._generate_recommendations(assessment)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error in quality assessment: {e}")
            return {
                "overall_score": 0.0,
                "error": str(e),
                "recommendations": ["Quality assessment failed - check extraction results format"]
            }
    
    def _assess_text_quality(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of extracted text."""
        text_content = self._extract_all_text(extraction_result)
        
        if not text_content:
            return {
                "score": 0.0,
                "metrics": {
                    "total_characters": 0,
                    "word_count": 0,
                    "readability_score": 0.0,
                    "language_coherence": 0.0,
                    "ocr_artifacts": 1.0
                }
            }
        
        metrics = {
            "total_characters": len(text_content),
            "word_count": len(text_content.split()),
            "readability_score": self._calculate_readability(text_content),
            "language_coherence": self._assess_language_coherence(text_content),
            "ocr_artifacts": self._detect_ocr_artifacts(text_content),
            "text_density": self._calculate_text_density(text_content),
            "character_distribution": self._analyze_character_distribution(text_content)
        }
        
        # Calculate text quality score (0-1)
        score = (
            min(metrics["readability_score"] / 100, 1.0) * 0.3 +
            metrics["language_coherence"] * 0.3 +
            (1 - metrics["ocr_artifacts"]) * 0.2 +
            min(metrics["text_density"], 1.0) * 0.2
        )
        
        return {
            "score": score,
            "metrics": metrics
        }
    
    def _assess_structure_quality(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of document structure extraction."""
        structure_metrics = {
            "has_tables": "tables" in extraction_result and len(extraction_result.get("tables", [])) > 0,
            "has_forms": "forms" in extraction_result and len(extraction_result.get("forms", [])) > 0,
            "has_images": "images" in extraction_result and len(extraction_result.get("images", [])) > 0,
            "has_hierarchy": self._check_hierarchical_structure(extraction_result),
            "table_quality": self._assess_table_quality(extraction_result.get("tables", [])),
            "form_quality": self._assess_form_quality(extraction_result.get("forms", [])),
            "layout_coherence": self._assess_layout_coherence(extraction_result)
        }
        
        # Calculate structure score
        structure_elements = sum([
            structure_metrics["has_tables"],
            structure_metrics["has_forms"], 
            structure_metrics["has_images"],
            structure_metrics["has_hierarchy"]
        ])
        
        quality_scores = [
            structure_metrics["table_quality"],
            structure_metrics["form_quality"],
            structure_metrics["layout_coherence"]
        ]
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        structure_score = (structure_elements / 4) * 0.4 + avg_quality * 0.6
        
        return {
            "score": structure_score,
            "metrics": structure_metrics
        }
    
    def _assess_confidence_metrics(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess confidence-related metrics."""
        confidence_scores = []
        
        # Collect confidence scores from different extraction engines
        if "text" in extraction_result and isinstance(extraction_result["text"], dict):
            if "confidence" in extraction_result["text"]:
                confidence_scores.append(extraction_result["text"]["confidence"])
        
        # Table confidence scores
        for table in extraction_result.get("tables", []):
            if isinstance(table, dict) and "confidence" in table:
                confidence_scores.append(table["confidence"])
        
        # Form confidence scores
        for form in extraction_result.get("forms", []):
            if isinstance(form, dict) and "confidence" in form:
                confidence_scores.append(form["confidence"])
        
        if not confidence_scores:
            return {
                "average_confidence": 0.5,  # Default when no confidence available
                "confidence_variance": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "confidence_distribution": "unknown"
            }
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        variance = sum((x - avg_confidence) ** 2 for x in confidence_scores) / len(confidence_scores)
        
        return {
            "average_confidence": avg_confidence,
            "confidence_variance": variance,
            "min_confidence": min(confidence_scores),
            "max_confidence": max(confidence_scores),
            "confidence_distribution": self._classify_confidence_distribution(confidence_scores)
        }
    
    def _assess_completeness(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess how complete the extraction appears to be."""
        completeness_indicators = {
            "has_text_content": bool(self._extract_all_text(extraction_result)),
            "has_structured_data": bool(extraction_result.get("tables") or extraction_result.get("forms")),
            "has_metadata": bool(extraction_result.get("metadata") or extraction_result.get("document_info")),
            "extraction_coverage": self._estimate_extraction_coverage(extraction_result)
        }
        
        # Calculate completeness score
        basic_completeness = sum([
            completeness_indicators["has_text_content"],
            completeness_indicators["has_structured_data"],
            completeness_indicators["has_metadata"]
        ]) / 3
        
        overall_completeness = (basic_completeness * 0.7 + 
                              completeness_indicators["extraction_coverage"] * 0.3)
        
        return {
            "score": overall_completeness,
            "indicators": completeness_indicators
        }
    
    def _assess_consistency(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess internal consistency of extraction results."""
        consistency_checks = {
            "text_encoding_consistent": self._check_text_encoding_consistency(extraction_result),
            "coordinate_system_consistent": self._check_coordinate_consistency(extraction_result),
            "format_consistency": self._check_format_consistency(extraction_result),
            "cross_reference_validity": self._check_cross_references(extraction_result)
        }
        
        consistency_score = sum(consistency_checks.values()) / len(consistency_checks)
        
        return {
            "score": consistency_score,
            "checks": consistency_checks
        }
    
    def _assess_image_quality(self, image: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """Assess the quality of the original document image."""
        try:
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                return {"score": 0.5, "error": "Unsupported image format"}
            
            # Convert to numpy array for analysis
            img_array = np.array(pil_image.convert('L'))  # Convert to grayscale
            
            metrics = {
                "resolution": pil_image.size,
                "aspect_ratio": pil_image.size[0] / pil_image.size[1],
                "brightness": float(np.mean(img_array)),
                "contrast": float(np.std(img_array)),
                "sharpness": self._calculate_sharpness(img_array),
                "noise_level": self._estimate_noise_level(img_array)
            }
            
            # Calculate image quality score
            # Normalize metrics to 0-1 scale
            brightness_score = 1 - abs(metrics["brightness"] - 128) / 128  # Optimal around 128
            contrast_score = min(metrics["contrast"] / 50, 1.0)  # Good contrast > 50
            sharpness_score = min(metrics["sharpness"] / 100, 1.0)
            noise_score = max(0, 1 - metrics["noise_level"] / 50)  # Lower noise is better
            
            overall_score = (brightness_score + contrast_score + sharpness_score + noise_score) / 4
            
            return {
                "score": overall_score,
                "metrics": metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing image quality: {e}")
            return {"score": 0.5, "error": str(e)}
    
    def _extract_all_text(self, extraction_result: Dict[str, Any]) -> str:
        """Extract all text content from extraction results."""
        text_parts = []
        
        # Main text content
        if "text" in extraction_result:
            if isinstance(extraction_result["text"], str):
                text_parts.append(extraction_result["text"])
            elif isinstance(extraction_result["text"], dict) and "content" in extraction_result["text"]:
                text_parts.append(extraction_result["text"]["content"])
        
        # Table text
        for table in extraction_result.get("tables", []):
            if isinstance(table, dict) and "data" in table:
                # Extract text from table data
                table_text = self._extract_table_text(table["data"])
                text_parts.append(table_text)
        
        # Form text
        for form in extraction_result.get("forms", []):
            if isinstance(form, dict) and "fields" in form:
                form_text = self._extract_form_text(form["fields"])
                text_parts.append(form_text)
        
        return " ".join(text_parts)
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (Flesch Reading Ease)."""
        if not text or len(text.split()) < 10:
            return 0.0
        
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = self._count_syllables(text)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * words / sentences) - (84.6 * syllables / words)
        return max(0, min(100, score))
    
    def _count_syllables(self, text: str) -> int:
        """Estimate syllable count in text."""
        # Simple syllable counting heuristic
        vowels = "aeiouyAEIOUY"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in text:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Adjust for silent 'e'
        if text.endswith('e') or text.endswith('E'):
            syllable_count -= 1
        
        return max(1, syllable_count)  # At least 1 syllable per word
    
    def _assess_language_coherence(self, text: str) -> float:
        """Assess how coherent the language appears."""
        if not text:
            return 0.0
        
        # Simple heuristics for language coherence
        words = text.split()
        if len(words) < 5:
            return 0.5
        
        # Check for reasonable word length distribution
        word_lengths = [len(word) for word in words]
        avg_word_length = sum(word_lengths) / len(word_lengths)
        
        # Check for reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        space_ratio = sum(c.isspace() for c in text) / len(text)
        
        # Coherence score based on various factors
        length_score = 1.0 if 3 <= avg_word_length <= 8 else 0.5
        alpha_score = min(alpha_ratio / 0.8, 1.0)  # Expect ~80% alphabetic
        space_score = min(space_ratio / 0.15, 1.0)  # Expect ~15% spaces
        
        return (length_score + alpha_score + space_score) / 3
    
    def _detect_ocr_artifacts(self, text: str) -> float:
        """Detect common OCR artifacts and return artifact ratio."""
        if not text:
            return 0.0
        
        # Common OCR artifacts
        artifacts = [
            r'[|]{2,}',  # Multiple pipes
            r'[_]{3,}',  # Multiple underscores
            r'[.]{3,}',  # Multiple dots
            r'[?]{2,}',  # Multiple question marks
            r'\b[a-z]{1}\b',  # Single letters (often OCR errors)
            r'[0-9][a-zA-Z][0-9]',  # Number-letter-number patterns
        ]
        
        artifact_count = 0
        for pattern in artifacts:
            artifact_count += len(re.findall(pattern, text))
        
        # Return ratio of artifacts to total characters
        return min(artifact_count / len(text), 1.0) if text else 0.0

    def _calculate_text_density(self, text: str) -> float:
        """Calculate text density (meaningful content ratio)."""
        if not text:
            return 0.0

        # Remove extra whitespace and count meaningful characters
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        meaningful_chars = sum(c.isalnum() for c in cleaned_text)

        return meaningful_chars / len(text) if text else 0.0

    def _analyze_character_distribution(self, text: str) -> Dict[str, float]:
        """Analyze character distribution in text."""
        if not text:
            return {"alphabetic": 0.0, "numeric": 0.0, "punctuation": 0.0, "whitespace": 0.0}

        total_chars = len(text)
        return {
            "alphabetic": sum(c.isalpha() for c in text) / total_chars,
            "numeric": sum(c.isdigit() for c in text) / total_chars,
            "punctuation": sum(c in ".,!?;:" for c in text) / total_chars,
            "whitespace": sum(c.isspace() for c in text) / total_chars
        }

    def _check_hierarchical_structure(self, extraction_result: Dict[str, Any]) -> bool:
        """Check if document has hierarchical structure."""
        # Look for structure indicators
        structure_indicators = [
            "hierarchy" in extraction_result,
            "structure" in extraction_result,
            "headings" in extraction_result,
            "sections" in extraction_result
        ]
        return any(structure_indicators)

    def _assess_table_quality(self, tables: List[Dict[str, Any]]) -> float:
        """Assess quality of extracted tables."""
        if not tables:
            return 0.0

        quality_scores = []
        for table in tables:
            score = 0.0

            # Check for proper structure
            if "data" in table or "cells" in table:
                score += 0.3

            # Check for headers
            if "headers" in table or "columns" in table:
                score += 0.2

            # Check for confidence
            if "confidence" in table and table["confidence"] > 0.7:
                score += 0.3

            # Check for reasonable dimensions
            if "shape" in table:
                rows, cols = table["shape"]
                if rows > 1 and cols > 1:
                    score += 0.2

            quality_scores.append(score)

        return sum(quality_scores) / len(quality_scores)

    def _assess_form_quality(self, forms: List[Dict[str, Any]]) -> float:
        """Assess quality of extracted forms."""
        if not forms:
            return 0.0

        quality_scores = []
        for form in forms:
            score = 0.0

            # Check for fields
            if "fields" in form and form["fields"]:
                score += 0.4

            # Check for field types
            if "field_types" in form:
                score += 0.2

            # Check for confidence
            if "confidence" in form and form["confidence"] > 0.7:
                score += 0.4

            quality_scores.append(score)

        return sum(quality_scores) / len(quality_scores)

    def _assess_layout_coherence(self, extraction_result: Dict[str, Any]) -> float:
        """Assess layout coherence."""
        # Simple heuristic based on presence of layout information
        layout_indicators = [
            "bbox" in str(extraction_result),
            "coordinates" in str(extraction_result),
            "layout" in extraction_result,
            "regions" in extraction_result
        ]
        return sum(layout_indicators) / len(layout_indicators)

    def _classify_confidence_distribution(self, confidence_scores: List[float]) -> str:
        """Classify the distribution of confidence scores."""
        if not confidence_scores:
            return "unknown"

        avg_conf = sum(confidence_scores) / len(confidence_scores)
        variance = sum((x - avg_conf) ** 2 for x in confidence_scores) / len(confidence_scores)

        if avg_conf > 0.8 and variance < 0.05:
            return "high_consistent"
        elif avg_conf > 0.6 and variance < 0.1:
            return "good_consistent"
        elif variance > 0.2:
            return "highly_variable"
        elif avg_conf < 0.4:
            return "low_confidence"
        else:
            return "moderate"

    def _estimate_extraction_coverage(self, extraction_result: Dict[str, Any]) -> float:
        """Estimate how much of the document was successfully extracted."""
        # Heuristic based on content richness
        coverage_factors = []

        # Text coverage
        text_content = self._extract_all_text(extraction_result)
        if text_content:
            # Assume good coverage if we have substantial text
            text_coverage = min(len(text_content) / 1000, 1.0)  # Normalize to 1000 chars
            coverage_factors.append(text_coverage)

        # Structural element coverage
        structural_elements = sum([
            len(extraction_result.get("tables", [])),
            len(extraction_result.get("forms", [])),
            len(extraction_result.get("images", []))
        ])

        if structural_elements > 0:
            coverage_factors.append(min(structural_elements / 5, 1.0))  # Normalize to 5 elements

        return sum(coverage_factors) / len(coverage_factors) if coverage_factors else 0.5

    def _check_text_encoding_consistency(self, extraction_result: Dict[str, Any]) -> bool:
        """Check if text encoding is consistent throughout."""
        text_content = self._extract_all_text(extraction_result)
        if not text_content:
            return True

        # Check for encoding issues
        encoding_issues = [
            '�',  # Replacement character
            '\ufffd',  # Unicode replacement character
            re.search(r'[^\x00-\x7F\u00A0-\uFFFF]', text_content)  # Invalid Unicode
        ]

        return not any(encoding_issues)

    def _check_coordinate_consistency(self, extraction_result: Dict[str, Any]) -> bool:
        """Check if coordinate systems are consistent."""
        # Look for bbox/coordinate information and check consistency
        coordinates = []

        # Extract coordinates from various sources
        for table in extraction_result.get("tables", []):
            if "bbox" in table:
                coordinates.append(table["bbox"])

        for form in extraction_result.get("forms", []):
            if "bbox" in form:
                coordinates.append(form["bbox"])

        if not coordinates:
            return True  # No coordinates to check

        # Check if all coordinates follow same format
        coord_formats = [len(coord) if isinstance(coord, (list, tuple)) else 0 for coord in coordinates]
        return len(set(coord_formats)) <= 1  # All should have same format

    def _check_format_consistency(self, extraction_result: Dict[str, Any]) -> bool:
        """Check if data formats are consistent."""
        # Check if similar data types have consistent formats
        consistency_checks = []

        # Check table format consistency
        tables = extraction_result.get("tables", [])
        if len(tables) > 1:
            table_keys = [set(table.keys()) for table in tables if isinstance(table, dict)]
            if table_keys:
                # Check if tables have similar structure
                common_keys = set.intersection(*table_keys) if table_keys else set()
                consistency_checks.append(len(common_keys) > 0)

        # Check form format consistency
        forms = extraction_result.get("forms", [])
        if len(forms) > 1:
            form_keys = [set(form.keys()) for form in forms if isinstance(form, dict)]
            if form_keys:
                common_keys = set.intersection(*form_keys) if form_keys else set()
                consistency_checks.append(len(common_keys) > 0)

        return all(consistency_checks) if consistency_checks else True

    def _check_cross_references(self, extraction_result: Dict[str, Any]) -> bool:
        """Check validity of cross-references within the document."""
        # Simple check - assume valid if no obvious inconsistencies
        return True  # Placeholder for more sophisticated cross-reference checking

    def _calculate_sharpness(self, img_array: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        try:
            if self._scipy_available:
                from scipy import ndimage
                laplacian = ndimage.laplace(img_array)
                return float(np.var(laplacian))
            else:
                # Simple gradient-based sharpness
                grad_x = np.diff(img_array, axis=1)
                grad_y = np.diff(img_array, axis=0)
                return float(np.mean(grad_x**2) + np.mean(grad_y**2))
        except Exception:
            return 50.0  # Default moderate sharpness

    def _estimate_noise_level(self, img_array: np.ndarray) -> float:
        """Estimate noise level in image."""
        try:
            # Use standard deviation of high-frequency components as noise estimate
            if img_array.shape[0] > 10 and img_array.shape[1] > 10:
                # Sample small patches and calculate variance
                patches = []
                for i in range(0, img_array.shape[0]-5, 10):
                    for j in range(0, img_array.shape[1]-5, 10):
                        patch = img_array[i:i+5, j:j+5]
                        patches.append(np.std(patch))

                return float(np.mean(patches)) if patches else 10.0
            else:
                return float(np.std(img_array))
        except Exception:
            return 10.0  # Default moderate noise

    def _extract_table_text(self, table_data: Any) -> str:
        """Extract text content from table data."""
        if isinstance(table_data, list):
            text_parts = []
            for row in table_data:
                if isinstance(row, list):
                    text_parts.extend([str(cell) for cell in row])
                elif isinstance(row, dict):
                    text_parts.extend([str(v) for v in row.values()])
            return " ".join(text_parts)
        return str(table_data)

    def _extract_form_text(self, form_fields: Any) -> str:
        """Extract text content from form fields."""
        if isinstance(form_fields, dict):
            return " ".join([str(v) for v in form_fields.values()])
        elif isinstance(form_fields, list):
            text_parts = []
            for field in form_fields:
                if isinstance(field, dict):
                    text_parts.extend([str(v) for v in field.values()])
                else:
                    text_parts.append(str(field))
            return " ".join(text_parts)
        return str(form_fields)

    def _calculate_overall_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate overall quality score from individual assessments."""
        scores = []
        weights = []

        # Text quality (weight: 0.3)
        if "text_quality" in assessment and "score" in assessment["text_quality"]:
            scores.append(assessment["text_quality"]["score"])
            weights.append(0.3)

        # Structure quality (weight: 0.25)
        if "structure_quality" in assessment and "score" in assessment["structure_quality"]:
            scores.append(assessment["structure_quality"]["score"])
            weights.append(0.25)

        # Confidence metrics (weight: 0.2)
        if "confidence_metrics" in assessment:
            conf_score = assessment["confidence_metrics"].get("average_confidence", 0.5)
            scores.append(conf_score)
            weights.append(0.2)

        # Completeness (weight: 0.15)
        if "completeness" in assessment and "score" in assessment["completeness"]:
            scores.append(assessment["completeness"]["score"])
            weights.append(0.15)

        # Consistency (weight: 0.1)
        if "consistency" in assessment and "score" in assessment["consistency"]:
            scores.append(assessment["consistency"]["score"])
            weights.append(0.1)

        if not scores:
            return 0.0

        # Weighted average
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on assessment results."""
        recommendations = []

        # Text quality recommendations
        if "text_quality" in assessment:
            text_score = assessment["text_quality"]["score"]
            if text_score < 0.5:
                recommendations.append("Consider using advanced OCR engines (TrOCR) for better text recognition")

            metrics = assessment["text_quality"].get("metrics", {})
            if metrics.get("ocr_artifacts", 0) > 0.1:
                recommendations.append("High OCR artifacts detected - consider image preprocessing")

        # Structure quality recommendations
        if "structure_quality" in assessment:
            struct_score = assessment["structure_quality"]["score"]
            if struct_score < 0.5:
                recommendations.append("Consider using Table Transformer for better structure extraction")

        # Confidence recommendations
        if "confidence_metrics" in assessment:
            avg_conf = assessment["confidence_metrics"].get("average_confidence", 0.5)
            if avg_conf < 0.6:
                recommendations.append("Low confidence scores - consider manual review of results")

        # Image quality recommendations
        if "image_quality" in assessment:
            img_score = assessment["image_quality"]["score"]
            if img_score < 0.5:
                recommendations.append("Poor image quality detected - consider image enhancement")

        # Overall score recommendations
        overall_score = assessment.get("overall_score", 0.0)
        if overall_score < 0.4:
            recommendations.append("Overall quality is low - consider using advanced extraction features")
        elif overall_score > 0.8:
            recommendations.append("High quality extraction - results are reliable")

        return recommendations if recommendations else ["Extraction quality appears acceptable"]


# Convenience function for easy usage
def assess_document_quality(
    extraction_result: Dict[str, Any],
    original_image: Optional[Union[str, Path, Image.Image]] = None
) -> Dict[str, Any]:
    """
    Convenience function to assess document extraction quality.

    Args:
        extraction_result: The result from document extraction
        original_image: Original document image (optional)

    Returns:
        Quality assessment result dictionary
    """
    assessor = DocumentQualityAssessor()
    return assessor.assess_extraction_quality(extraction_result, original_image)
