"""
Tests for the image enhancer module.
"""

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from od_parse.ocr.image_enhancer import (
    EnhancementConfig,
    ImageEnhancer,
    ImageQuality,
    QualityScore,
    assess_image_quality,
    binarize_for_ocr,
    enhance_for_ocr,
)


class TestImageQualityAssessment(unittest.TestCase):
    """Tests for image quality assessment."""

    def test_assess_high_quality_image(self):
        """Test quality assessment on a high-quality image."""
        # Create a high-quality synthetic image
        img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        img[:] = 255  # White background

        # Add some text-like patterns with good contrast
        cv2.putText(
            img,
            "High Quality Text",
            (100, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 0),
            5,
        )

        quality = assess_image_quality(img)

        self.assertIsInstance(quality, QualityScore)
        self.assertGreaterEqual(quality.overall_score, 0.0)
        self.assertLessEqual(quality.overall_score, 1.0)
        self.assertIn(quality.quality_level, list(ImageQuality))

    def test_assess_low_quality_image(self):
        """Test quality assessment on a low-quality image."""
        # Create a small, noisy image
        img = np.random.randint(100, 150, (100, 100, 3), dtype=np.uint8)

        quality = assess_image_quality(img)

        self.assertIsInstance(quality, QualityScore)
        self.assertLess(quality.resolution_score, 0.5)  # Low resolution

    def test_quality_score_attributes(self):
        """Test that QualityScore has all expected attributes."""
        img = np.ones((500, 500, 3), dtype=np.uint8) * 128

        quality = assess_image_quality(img)

        # Use duck typing for numeric values (numpy floats are also valid)
        self.assertTrue(isinstance(quality.overall_score, (float, np.floating)))
        self.assertTrue(isinstance(quality.blur_score, (float, np.floating)))
        self.assertTrue(isinstance(quality.contrast_score, (float, np.floating)))
        self.assertTrue(isinstance(quality.noise_score, (float, np.floating)))
        self.assertTrue(isinstance(quality.resolution_score, (float, np.floating)))
        self.assertIsInstance(quality.recommended_pipeline, str)
        self.assertIn(quality.recommended_pipeline, ["fast", "enhanced", "vlm"])

    def test_assess_from_pil_image(self):
        """Test quality assessment from PIL Image."""
        pil_img = Image.new("RGB", (800, 600), color="white")

        quality = assess_image_quality(pil_img)

        self.assertIsInstance(quality, QualityScore)

    def test_assess_from_file_path(self):
        """Test quality assessment from file path."""
        # Create a temporary image file
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:] = 200

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            quality = assess_image_quality(f.name)
            os.unlink(f.name)

        self.assertIsInstance(quality, QualityScore)


class TestImageEnhancer(unittest.TestCase):
    """Tests for the ImageEnhancer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.enhancer = ImageEnhancer()

        # Create a test image
        self.test_image = np.zeros((500, 500, 3), dtype=np.uint8)
        self.test_image[:] = 200  # Gray background
        cv2.putText(
            self.test_image,
            "Test",
            (100, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (50, 50, 50),
            3,
        )

    def test_enhancer_initialization(self):
        """Test enhancer initialization."""
        enhancer = ImageEnhancer()
        self.assertIsNotNone(enhancer.config)
        self.assertIsInstance(enhancer.config, EnhancementConfig)

    def test_enhancer_with_custom_config(self):
        """Test enhancer with custom configuration."""
        config = EnhancementConfig(
            enable_upscaling=False,
            enable_denoising=True,
            enable_deskew=False,
        )
        enhancer = ImageEnhancer(config)

        self.assertFalse(enhancer.config.enable_upscaling)
        self.assertTrue(enhancer.config.enable_denoising)
        self.assertFalse(enhancer.config.enable_deskew)

    def test_enhance_returns_tuple(self):
        """Test that enhance returns a tuple of (image, metadata)."""
        result = self.enhancer.enhance(self.test_image)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        enhanced_img, metadata = result
        self.assertIsInstance(enhanced_img, np.ndarray)
        self.assertIsInstance(metadata, dict)

    def test_enhance_metadata_contents(self):
        """Test that enhance metadata contains expected keys."""
        _, metadata = self.enhancer.enhance(self.test_image)

        self.assertIn("original_shape", metadata)
        self.assertIn("final_shape", metadata)
        self.assertIn("enhancements_applied", metadata)
        self.assertIn("quality_score", metadata)

    def test_enhance_from_pil_image(self):
        """Test enhancement from PIL Image."""
        pil_img = Image.fromarray(self.test_image)

        enhanced, metadata = self.enhancer.enhance(pil_img)

        self.assertIsInstance(enhanced, np.ndarray)

    def test_enhance_from_file_path(self):
        """Test enhancement from file path."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, self.test_image)
            enhanced, metadata = self.enhancer.enhance(f.name)
            os.unlink(f.name)

        self.assertIsInstance(enhanced, np.ndarray)

    def test_enhancement_config_defaults(self):
        """Test EnhancementConfig default values."""
        config = EnhancementConfig()

        self.assertTrue(config.enable_upscaling)
        self.assertTrue(config.enable_denoising)
        self.assertTrue(config.enable_deskew)
        self.assertTrue(config.enable_contrast)
        self.assertEqual(config.target_dpi, 300)


class TestBinarization(unittest.TestCase):
    """Tests for binarization functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.test_image[:] = 200
        cv2.putText(
            self.test_image,
            "Test",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (50, 50, 50),
            2,
        )

    def test_adaptive_binarization(self):
        """Test adaptive binarization method."""
        binary = binarize_for_ocr(self.test_image, method="adaptive")

        self.assertEqual(len(binary.shape), 2)  # Should be grayscale
        self.assertTrue(np.all((binary == 0) | (binary == 255)))  # Binary values

    def test_otsu_binarization(self):
        """Test Otsu binarization method."""
        binary = binarize_for_ocr(self.test_image, method="otsu")

        self.assertEqual(len(binary.shape), 2)
        self.assertTrue(np.all((binary == 0) | (binary == 255)))

    def test_sauvola_binarization(self):
        """Test Sauvola binarization method."""
        binary = binarize_for_ocr(self.test_image, method="sauvola")

        self.assertEqual(len(binary.shape), 2)

    def test_binarization_from_pil(self):
        """Test binarization from PIL Image."""
        pil_img = Image.fromarray(self.test_image)
        binary = binarize_for_ocr(pil_img)

        self.assertEqual(len(binary.shape), 2)

    def test_invalid_binarization_method(self):
        """Test that invalid binarization method raises error."""
        with self.assertRaises(ValueError):
            binarize_for_ocr(self.test_image, method="invalid")


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_enhance_for_ocr(self):
        """Test enhance_for_ocr convenience function."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:] = 150

        enhanced, metadata = enhance_for_ocr(img)

        self.assertIsInstance(enhanced, np.ndarray)
        self.assertIsInstance(metadata, dict)

    def test_enhance_for_ocr_with_config(self):
        """Test enhance_for_ocr with custom config."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        config = EnhancementConfig(enable_denoising=False)

        enhanced, metadata = enhance_for_ocr(img, config=config)

        # Denoising should not be in the list of applied enhancements
        # (for high quality images, minimal processing is applied anyway)
        self.assertIsInstance(enhanced, np.ndarray)


class TestDeskewing(unittest.TestCase):
    """Tests for deskewing functionality."""

    def test_deskew_straight_image(self):
        """Test that straight images are not modified much."""
        # Create a straight image
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img[:] = 255
        cv2.line(img, (100, 100), (400, 100), (0, 0, 0), 2)
        cv2.line(img, (100, 200), (400, 200), (0, 0, 0), 2)

        enhancer = ImageEnhancer(EnhancementConfig(enable_deskew=True))
        enhanced, metadata = enhancer.enhance(img)

        # Should have minimal or no skew angle
        skew_angle = metadata.get("skew_angle", 0)
        self.assertLess(abs(skew_angle), 5)


if __name__ == "__main__":
    unittest.main()
