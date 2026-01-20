"""
Advanced Image Enhancement for OCR.

This module provides preprocessing functions to improve OCR accuracy on
low-quality images including:
- Super-resolution upscaling
- Adaptive denoising
- Automatic deskewing
- Dynamic contrast enhancement
- Background/shadow removal
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ImageQuality(Enum):
    """Image quality levels for routing."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class QualityScore:
    """Image quality assessment result."""

    overall_score: float  # 0-1, higher is better
    quality_level: ImageQuality
    blur_score: float  # 0-1, higher means less blur
    contrast_score: float  # 0-1, higher is better contrast
    noise_score: float  # 0-1, higher means less noise
    resolution_score: float  # 0-1, based on image dimensions
    recommended_pipeline: str  # 'fast', 'enhanced', or 'vlm'
    details: Dict[str, Any]


@dataclass
class EnhancementConfig:
    """Configuration for image enhancement."""

    # Upscaling
    enable_upscaling: bool = True
    target_dpi: int = 300
    min_dimension: int = 1000
    max_upscale_factor: float = 4.0

    # Denoising
    enable_denoising: bool = True
    denoise_strength: int = 10
    denoise_color_strength: int = 10

    # Deskewing
    enable_deskew: bool = True
    max_skew_angle: float = 45.0

    # Contrast enhancement
    enable_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)

    # Background removal
    enable_background_removal: bool = True

    # Shadow removal
    enable_shadow_removal: bool = True


class ImageEnhancer:
    """
    Advanced image enhancer for OCR preprocessing.

    Applies multiple enhancement techniques to improve OCR accuracy
    on low-quality images.
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        """
        Initialize the image enhancer.

        Args:
            config: Enhancement configuration. Uses defaults if not provided.
        """
        self.config = config or EnhancementConfig()
        self.logger = get_logger(__name__)

    def enhance(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        quality_score: Optional[QualityScore] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply all enabled enhancements to an image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            quality_score: Optional pre-computed quality score

        Returns:
            Tuple of (enhanced image as numpy array, enhancement metadata)
        """
        # Load image
        img = self._load_image(image)
        original_shape = img.shape[:2]
        metadata = {
            "original_shape": original_shape,
            "enhancements_applied": [],
        }

        # Assess quality if not provided
        if quality_score is None:
            quality_score = assess_image_quality(img)
        metadata["quality_score"] = quality_score.overall_score

        # Apply enhancements based on quality
        if quality_score.quality_level == ImageQuality.HIGH:
            # Minimal processing for high-quality images
            if self.config.enable_contrast:
                img = self._apply_light_contrast(img)
                metadata["enhancements_applied"].append("light_contrast")
        else:
            # Full enhancement pipeline for medium/low quality
            if self.config.enable_shadow_removal:
                img = self._remove_shadows(img)
                metadata["enhancements_applied"].append("shadow_removal")

            if self.config.enable_background_removal:
                img = self._normalize_background(img)
                metadata["enhancements_applied"].append("background_normalization")

            if self.config.enable_deskew:
                img, angle = self._deskew(img)
                metadata["enhancements_applied"].append("deskew")
                metadata["skew_angle"] = angle

            if self.config.enable_upscaling:
                img = self._upscale(img)
                metadata["enhancements_applied"].append("upscaling")

            if self.config.enable_denoising:
                img = self._denoise(img)
                metadata["enhancements_applied"].append("denoising")

            if self.config.enable_contrast:
                img = self._enhance_contrast(img)
                metadata["enhancements_applied"].append("contrast_enhancement")

        metadata["final_shape"] = img.shape[:2]
        self.logger.info(
            f"Image enhanced: {metadata['enhancements_applied']}, "
            f"shape {original_shape} -> {img.shape[:2]}"
        )

        return img, metadata

    def _load_image(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """Load and convert image to numpy array."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            return img
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            return image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        """
        Upscale low-resolution images using INTER_CUBIC interpolation.

        For very low resolution, applies super-resolution if available.
        """
        height, width = img.shape[:2]
        min_dim = min(height, width)

        if min_dim >= self.config.min_dimension:
            return img

        # Calculate scale factor
        scale = min(
            self.config.max_upscale_factor,
            self.config.min_dimension / min_dim,
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        self.logger.info(
            f"Upscaling image from {width}x{height} to {new_width}x{new_height}"
        )

        # Use INTER_CUBIC for better quality upscaling
        upscaled = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_CUBIC
        )

        # Apply light sharpening to compensate for upscale blur
        kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)

        # Blend original upscale with sharpened version
        return cv2.addWeighted(upscaled, 0.5, sharpened, 0.5, 0)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Apply adaptive denoising using non-local means.

        Uses fastNlMeansDenoisingColored for color images,
        fastNlMeansDenoising for grayscale.
        """
        if len(img.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                img,
                None,
                self.config.denoise_strength,
                self.config.denoise_color_strength,
                7,
                21,
            )
        else:
            return cv2.fastNlMeansDenoising(
                img, None, self.config.denoise_strength, 7, 21
            )

    def _deskew(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Automatically detect and correct skew angle.

        Uses projection profile method for text documents.
        """
        # Convert to grayscale for angle detection
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find coordinates of non-zero pixels
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) < 100:
            # Not enough pixels to determine angle
            return img, 0.0

        # Calculate skew angle using minAreaRect
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust angle to -45 to 45 range
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        # Only correct if angle is significant
        if abs(angle) < 0.5:
            return img, 0.0

        if abs(angle) > self.config.max_skew_angle:
            self.logger.warning(
                f"Skew angle {angle:.1f}° exceeds max {self.config.max_skew_angle}°"
            )
            return img, angle

        self.logger.info(f"Correcting skew angle: {angle:.2f}°")

        # Rotate to correct skew
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image bounds
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix for translation
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            img,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated, angle

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Works on L channel for color images, directly on grayscale.
        """
        if len(img.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_size,
            )
            enhanced_l = clahe.apply(l_channel)

            # Merge and convert back
            enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_size,
            )
            return clahe.apply(img)

    def _apply_light_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply light contrast enhancement for high-quality images."""
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
            enhanced_l = clahe.apply(l_channel)
            enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
            return clahe.apply(img)

    def _remove_shadows(self, img: np.ndarray) -> np.ndarray:
        """
        Remove shadows from document images.

        Uses morphological operations to estimate and remove background shadows.
        """
        if len(img.shape) != 3:
            return img

        # Split into RGB channels
        rgb_planes = cv2.split(img)
        result_planes = []

        for plane in rgb_planes:
            # Estimate background using large morphological dilation
            dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            background = cv2.medianBlur(dilated, 21)

            # Compute difference and normalize
            diff = 255 - cv2.absdiff(plane, background)
            normalized = cv2.normalize(
                diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            result_planes.append(normalized)

        return cv2.merge(result_planes)

    def _normalize_background(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize background for camera-captured documents.

        Attempts to create uniform white background.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Estimate background using morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Normalize
        normalized = gray.astype(np.float32) / (background.astype(np.float32) + 1e-6)
        normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)

        if len(img.shape) == 3:
            # Convert back to color
            return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        return normalized


def assess_image_quality(
    image: Union[str, Path, Image.Image, np.ndarray]
) -> QualityScore:
    """
    Assess image quality and recommend processing pipeline.

    Args:
        image: Input image

    Returns:
        QualityScore with metrics and recommendations
    """
    # Load image
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Could not load image: {image}")
    elif isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Convert to grayscale for analysis
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    height, width = gray.shape[:2]
    details = {"width": width, "height": height}

    # 1. Blur detection using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(1.0, laplacian_var / 500.0)  # Normalize to 0-1
    details["laplacian_variance"] = laplacian_var

    # 2. Contrast assessment
    contrast = gray.std()
    contrast_score = min(1.0, contrast / 75.0)  # Normalize to 0-1
    details["contrast_std"] = contrast

    # 3. Noise estimation using local variance
    local_var = cv2.blur(
        (gray.astype(np.float32) - cv2.blur(gray, (5, 5)).astype(np.float32)) ** 2,
        (5, 5),
    )
    noise_level = np.sqrt(local_var.mean())
    noise_score = max(0.0, 1.0 - (noise_level / 30.0))  # Lower noise = higher score
    details["noise_level"] = noise_level

    # 4. Resolution score
    min_dim = min(width, height)
    if min_dim >= 2000:
        resolution_score = 1.0
    elif min_dim >= 1000:
        resolution_score = 0.8
    elif min_dim >= 500:
        resolution_score = 0.5
    else:
        resolution_score = max(0.2, min_dim / 500.0)
    details["min_dimension"] = min_dim

    # Calculate overall score (weighted average)
    overall_score = (
        blur_score * 0.35
        + contrast_score * 0.25
        + noise_score * 0.20
        + resolution_score * 0.20
    )

    # Determine quality level and recommended pipeline
    if overall_score >= 0.7:
        quality_level = ImageQuality.HIGH
        recommended_pipeline = "fast"
    elif overall_score >= 0.4:
        quality_level = ImageQuality.MEDIUM
        recommended_pipeline = "enhanced"
    else:
        quality_level = ImageQuality.LOW
        recommended_pipeline = "vlm"

    return QualityScore(
        overall_score=overall_score,
        quality_level=quality_level,
        blur_score=blur_score,
        contrast_score=contrast_score,
        noise_score=noise_score,
        resolution_score=resolution_score,
        recommended_pipeline=recommended_pipeline,
        details=details,
    )


def enhance_for_ocr(
    image: Union[str, Path, Image.Image, np.ndarray],
    config: Optional[EnhancementConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to enhance an image for OCR.

    Args:
        image: Input image
        config: Optional enhancement configuration

    Returns:
        Tuple of (enhanced image, metadata)
    """
    enhancer = ImageEnhancer(config)
    return enhancer.enhance(image)


def binarize_for_ocr(
    image: Union[np.ndarray, Image.Image],
    method: str = "adaptive",
) -> np.ndarray:
    """
    Convert image to binary (black and white) for OCR.

    Args:
        image: Input image
        method: 'adaptive', 'otsu', or 'sauvola'

    Returns:
        Binary image
    """
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "sauvola":
        # Sauvola binarization (local thresholding)
        window_size = 25
        k = 0.2
        mean = cv2.blur(gray.astype(np.float32), (window_size, window_size))
        mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (window_size, window_size))
        std = np.sqrt(mean_sq - mean**2)
        threshold = mean * (1 + k * (std / 128 - 1))
        binary = ((gray > threshold) * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return binary
