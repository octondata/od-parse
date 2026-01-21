"""
Tests for the multi-engine OCR module.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image

from od_parse.ocr.multi_engine import (
    DOCTR_AVAILABLE,
    EASYOCR_AVAILABLE,
    TESSERACT_AVAILABLE,
    DocTREngine,
    EasyOCREngine,
    MultiEngineOCR,
    OCRResult,
    TesseractEngine,
    get_available_ocr_engines,
)


class TestOCRResult(unittest.TestCase):
    """Tests for OCRResult dataclass."""

    def test_ocr_result_creation(self):
        """Test creating an OCRResult."""
        result = OCRResult(
            text="Hello World",
            confidence=0.95,
            engine="tesseract",
        )

        self.assertEqual(result.text, "Hello World")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.engine, "tesseract")
        self.assertEqual(result.bounding_boxes, [])
        self.assertEqual(result.metadata, {})
        self.assertIsNone(result.error)

    def test_ocr_result_with_error(self):
        """Test OCRResult with error."""
        result = OCRResult(
            text="",
            confidence=0.0,
            engine="failed_engine",
            error="Engine not available",
        )

        self.assertEqual(result.error, "Engine not available")

    def test_ocr_result_with_metadata(self):
        """Test OCRResult with metadata."""
        result = OCRResult(
            text="Test",
            confidence=0.8,
            engine="test",
            metadata={"processing_time": 1.5},
        )

        self.assertEqual(result.metadata["processing_time"], 1.5)


class TestEngineAvailability(unittest.TestCase):
    """Tests for engine availability detection."""

    def test_get_available_engines(self):
        """Test getting available OCR engines."""
        engines = get_available_ocr_engines()

        self.assertIsInstance(engines, dict)
        self.assertIn("easyocr", engines)
        self.assertIn("doctr", engines)
        self.assertIn("tesseract", engines)
        self.assertIn("trocr", engines)

        # Values should be booleans
        for engine, available in engines.items():
            self.assertIsInstance(available, bool)


class TestTesseractEngine(unittest.TestCase):
    """Tests for TesseractEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = TesseractEngine()

        # Create a test image with text
        self.test_image = np.zeros((200, 400, 3), dtype=np.uint8)
        self.test_image[:] = 255  # White background
        cv2.putText(
            self.test_image,
            "Test Text",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

    def test_engine_name(self):
        """Test engine name property."""
        self.assertEqual(self.engine.name, "tesseract")

    def test_engine_availability(self):
        """Test engine availability property."""
        self.assertEqual(self.engine.is_available, TESSERACT_AVAILABLE)

    @unittest.skipUnless(TESSERACT_AVAILABLE, "Tesseract not available")
    def test_extract_text_returns_ocr_result(self):
        """Test that extract_text returns OCRResult."""
        result = self.engine.extract_text(self.test_image)

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.engine, "tesseract")

    @unittest.skipUnless(TESSERACT_AVAILABLE, "Tesseract not available")
    def test_extract_text_from_pil_image(self):
        """Test extraction from PIL Image."""
        pil_img = Image.fromarray(self.test_image)
        result = self.engine.extract_text(pil_img)

        self.assertIsInstance(result, OCRResult)

    @unittest.skipUnless(TESSERACT_AVAILABLE, "Tesseract not available")
    def test_extract_text_from_file(self):
        """Test extraction from file path."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        # Write image after closing file handle (Windows compatibility)
        cv2.imwrite(temp_path, self.test_image)
        try:
            result = self.engine.extract_text(temp_path)
            self.assertIsInstance(result, OCRResult)
        finally:
            os.unlink(temp_path)

    def test_unavailable_engine_error(self):
        """Test behavior when engine is not available."""
        if not TESSERACT_AVAILABLE:
            result = self.engine.extract_text(self.test_image)
            self.assertIsNotNone(result.error)


class TestEasyOCREngine(unittest.TestCase):
    """Tests for EasyOCREngine."""

    def test_engine_name(self):
        """Test engine name property."""
        engine = EasyOCREngine()
        self.assertEqual(engine.name, "easyocr")

    def test_engine_availability(self):
        """Test engine availability property."""
        engine = EasyOCREngine()
        self.assertEqual(engine.is_available, EASYOCR_AVAILABLE)

    def test_default_languages(self):
        """Test default language configuration."""
        engine = EasyOCREngine()
        self.assertEqual(engine.languages, ["en"])

    def test_custom_languages(self):
        """Test custom language configuration."""
        engine = EasyOCREngine(languages=["en", "fr", "de"])
        self.assertEqual(engine.languages, ["en", "fr", "de"])

    def test_unavailable_engine_error(self):
        """Test behavior when engine is not available."""
        engine = EasyOCREngine()
        if not EASYOCR_AVAILABLE:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = engine.extract_text(img)
            self.assertIsNotNone(result.error)


class TestDocTREngine(unittest.TestCase):
    """Tests for DocTREngine."""

    def test_engine_name(self):
        """Test engine name property."""
        engine = DocTREngine()
        self.assertEqual(engine.name, "doctr")

    def test_engine_availability(self):
        """Test engine availability property."""
        engine = DocTREngine()
        self.assertEqual(engine.is_available, DOCTR_AVAILABLE)

    def test_default_architectures(self):
        """Test default architecture configuration."""
        engine = DocTREngine()
        self.assertEqual(engine.det_arch, "db_resnet50")
        self.assertEqual(engine.reco_arch, "crnn_vgg16_bn")

    def test_unavailable_engine_error(self):
        """Test behavior when engine is not available."""
        engine = DocTREngine()
        if not DOCTR_AVAILABLE:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = engine.extract_text(img)
            self.assertIsNotNone(result.error)


class TestMultiEngineOCR(unittest.TestCase):
    """Tests for MultiEngineOCR class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test image
        self.test_image = np.zeros((200, 400, 3), dtype=np.uint8)
        self.test_image[:] = 255
        cv2.putText(
            self.test_image,
            "Hello OCR",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

    def test_initialization_default(self):
        """Test default initialization."""
        processor = MultiEngineOCR()

        self.assertTrue(processor.enable_enhancement)
        self.assertTrue(processor.vlm_fallback)

    def test_initialization_custom_engines(self):
        """Test initialization with custom engines."""
        processor = MultiEngineOCR(engines=["tesseract"])

        # Should only have tesseract if available
        if TESSERACT_AVAILABLE:
            self.assertIn("tesseract", processor.available_engines)

    def test_available_engines_property(self):
        """Test available_engines property."""
        processor = MultiEngineOCR()

        engines = processor.available_engines
        self.assertIsInstance(engines, list)

    def test_extract_text_returns_ocr_result(self):
        """Test that extract_text returns OCRResult."""
        processor = MultiEngineOCR()

        if processor.available_engines:
            result = processor.extract_text(self.test_image)
            self.assertIsInstance(result, OCRResult)

    def test_extract_text_with_specific_engine(self):
        """Test extraction with specific engine."""
        processor = MultiEngineOCR()

        if "tesseract" in processor.available_engines:
            result = processor.extract_text(self.test_image, engine="tesseract")
            self.assertIn("tesseract", result.engine)

    def test_extract_text_with_unavailable_engine(self):
        """Test extraction with unavailable engine."""
        processor = MultiEngineOCR()

        result = processor.extract_text(self.test_image, engine="nonexistent")
        self.assertIsNotNone(result.error)

    def test_disable_enhancement(self):
        """Test with enhancement disabled."""
        processor = MultiEngineOCR(enable_enhancement=False)

        self.assertFalse(processor.enable_enhancement)
        self.assertIsNone(processor.enhancer)

    def test_disable_vlm_fallback(self):
        """Test with VLM fallback disabled."""
        processor = MultiEngineOCR(vlm_fallback=False)

        self.assertFalse(processor.vlm_fallback)

    def test_vlm_confidence_threshold(self):
        """Test VLM confidence threshold setting."""
        processor = MultiEngineOCR(vlm_confidence_threshold=0.5)

        self.assertEqual(processor.vlm_confidence_threshold, 0.5)


class TestEnsembleExtraction(unittest.TestCase):
    """Tests for ensemble OCR extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.zeros((200, 400, 3), dtype=np.uint8)
        self.test_image[:] = 255
        cv2.putText(
            self.test_image,
            "Ensemble Test",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

    def test_ensemble_extraction(self):
        """Test ensemble extraction."""
        processor = MultiEngineOCR()

        if len(processor.available_engines) >= 1:
            result = processor.extract_text(self.test_image, use_ensemble=True)
            self.assertIsInstance(result, OCRResult)

    def test_ensemble_metadata(self):
        """Test that ensemble extraction includes metadata."""
        processor = MultiEngineOCR()

        if len(processor.available_engines) >= 2:
            result = processor.extract_text(self.test_image, use_ensemble=True)
            if "ensemble" in result.engine:
                self.assertIn("ensemble_results", result.metadata)


class TestImageInputFormats(unittest.TestCase):
    """Tests for different image input formats."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = MultiEngineOCR()
        self.np_image = np.zeros((100, 200, 3), dtype=np.uint8)
        self.np_image[:] = 255

    def test_numpy_array_input(self):
        """Test extraction from numpy array."""
        if self.processor.available_engines:
            result = self.processor.extract_text(self.np_image)
            self.assertIsInstance(result, OCRResult)

    def test_pil_image_input(self):
        """Test extraction from PIL Image."""
        pil_img = Image.fromarray(self.np_image)

        if self.processor.available_engines:
            result = self.processor.extract_text(pil_img)
            self.assertIsInstance(result, OCRResult)

    def test_file_path_input(self):
        """Test extraction from file path."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        # Write image after closing file handle (Windows compatibility)
        cv2.imwrite(temp_path, self.np_image)
        try:
            if self.processor.available_engines:
                result = self.processor.extract_text(temp_path)
                self.assertIsInstance(result, OCRResult)
        finally:
            os.unlink(temp_path)

    def test_path_object_input(self):
        """Test extraction from Path object."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        # Write image after closing file handle (Windows compatibility)
        cv2.imwrite(temp_path, self.np_image)
        try:
            if self.processor.available_engines:
                result = self.processor.extract_text(Path(temp_path))
                self.assertIsInstance(result, OCRResult)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
