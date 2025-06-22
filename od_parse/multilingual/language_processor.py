"""
Multi-Language Processing Module

This module provides comprehensive multilingual support for document processing,
including language detection, text processing, and translation capabilities.
"""

import os
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

from od_parse.config import get_advanced_config
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MultilingualProcessor:
    """
    Comprehensive multilingual document processing engine.
    
    This class provides language detection, text processing, and translation
    capabilities for documents in multiple languages.
    """
    
    def __init__(self):
        """Initialize the multilingual processor."""
        self.logger = get_logger(__name__)
        
        # Available components
        self._langdetect_available = False
        self._spacy_available = False
        self._polyglot_available = False
        self._googletrans_available = False
        
        # Language models cache
        self._spacy_models = {}
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for multilingual processing dependencies."""
        config = get_advanced_config()
        
        if not config.is_feature_enabled("multilingual"):
            self.logger.info("Multilingual feature is disabled. Use config.enable_feature('multilingual') to enable.")
            return
        
        # Check langdetect
        try:
            import langdetect
            self._langdetect_available = True
            self.logger.info("langdetect available for language detection")
        except ImportError:
            self.logger.warning("langdetect not available. Language detection will be limited.")
        
        # Check spaCy
        try:
            import spacy
            self._spacy_available = True
            self.logger.info("spaCy available for text processing")
        except ImportError:
            self.logger.warning("spaCy not available. Advanced text processing will be limited.")
        
        # Check polyglot
        try:
            import polyglot
            self._polyglot_available = True
            self.logger.info("polyglot available for multilingual NLP")
        except ImportError:
            self.logger.warning("polyglot not available. Some multilingual features will be limited.")
        
        # Check googletrans
        try:
            import googletrans
            self._googletrans_available = True
            self.logger.info("googletrans available for translation")
        except ImportError:
            self.logger.warning("googletrans not available. Translation features will be limited.")
    
    def detect_language(self, text: str, method: str = "auto") -> Dict[str, Any]:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to analyze
            method: Detection method ('auto', 'langdetect', 'polyglot', 'heuristic')
            
        Returns:
            Dictionary containing language detection results
        """
        if not text or not text.strip():
            return {
                "language": "unknown",
                "confidence": 0.0,
                "method": method,
                "error": "Empty text"
            }
        
        try:
            if method == "auto":
                # Try methods in order of preference
                if self._langdetect_available:
                    return self._detect_with_langdetect(text)
                elif self._polyglot_available:
                    return self._detect_with_polyglot(text)
                else:
                    return self._detect_with_heuristics(text)
            
            elif method == "langdetect" and self._langdetect_available:
                return self._detect_with_langdetect(text)
            
            elif method == "polyglot" and self._polyglot_available:
                return self._detect_with_polyglot(text)
            
            elif method == "heuristic":
                return self._detect_with_heuristics(text)
            
            else:
                return {
                    "language": "unknown",
                    "confidence": 0.0,
                    "method": method,
                    "error": f"Method '{method}' not available"
                }
                
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return {
                "language": "unknown",
                "confidence": 0.0,
                "method": method,
                "error": str(e)
            }
    
    def _detect_with_langdetect(self, text: str) -> Dict[str, Any]:
        """Detect language using langdetect library."""
        try:
            from langdetect import detect, detect_langs
            
            # Get primary language
            primary_lang = detect(text)
            
            # Get confidence scores for all detected languages
            lang_probs = detect_langs(text)
            
            # Find confidence for primary language
            confidence = 0.0
            for lang_prob in lang_probs:
                if lang_prob.lang == primary_lang:
                    confidence = lang_prob.prob
                    break
            
            return {
                "language": primary_lang,
                "confidence": confidence,
                "method": "langdetect",
                "all_languages": [{"language": lp.lang, "confidence": lp.prob} for lp in lang_probs]
            }
            
        except Exception as e:
            self.logger.error(f"langdetect failed: {e}")
            return self._detect_with_heuristics(text)
    
    def _detect_with_polyglot(self, text: str) -> Dict[str, Any]:
        """Detect language using polyglot library."""
        try:
            from polyglot.detect import Detector
            
            detector = Detector(text)
            
            return {
                "language": detector.language.code,
                "confidence": detector.language.confidence,
                "method": "polyglot",
                "language_name": detector.language.name
            }
            
        except Exception as e:
            self.logger.error(f"polyglot detection failed: {e}")
            return self._detect_with_heuristics(text)
    
    def _detect_with_heuristics(self, text: str) -> Dict[str, Any]:
        """Detect language using simple heuristics."""
        # Simple character-based heuristics
        char_patterns = {
            'en': r'[a-zA-Z\s]',
            'es': r'[a-zA-ZñáéíóúüÑÁÉÍÓÚÜ\s]',
            'fr': r'[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ\s]',
            'de': r'[a-zA-ZäöüßÄÖÜ\s]',
            'it': r'[a-zA-ZàèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ\s]',
            'pt': r'[a-zA-ZáâãàéêíóôõúçÁÂÃÀÉÊÍÓÔÕÚÇ\s]',
            'ru': r'[а-яёА-ЯЁ\s]',
            'zh': r'[\u4e00-\u9fff]',
            'ja': r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]',
            'ko': r'[\uac00-\ud7af]',
            'ar': r'[\u0600-\u06ff]',
            'hi': r'[\u0900-\u097f]'
        }
        
        # Count matches for each language
        scores = {}
        total_chars = len(text)
        
        for lang, pattern in char_patterns.items():
            matches = len(re.findall(pattern, text))
            scores[lang] = matches / total_chars if total_chars > 0 else 0
        
        # Find best match
        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]
        
        # Default to English if no clear winner
        if best_score < 0.3:
            best_lang = "en"
            best_score = 0.5
        
        return {
            "language": best_lang,
            "confidence": best_score,
            "method": "heuristic",
            "all_scores": scores
        }
    
    def process_multilingual_text(
        self, 
        text: str, 
        target_language: Optional[str] = None,
        include_translation: bool = False
    ) -> Dict[str, Any]:
        """
        Process text with multilingual capabilities.
        
        Args:
            text: Text to process
            target_language: Target language for translation (optional)
            include_translation: Whether to include translation
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Detect language
            detection_result = self.detect_language(text)
            detected_lang = detection_result["language"]
            
            # Process text with appropriate language model
            processing_result = self._process_text_by_language(text, detected_lang)
            
            # Add translation if requested
            translation_result = {}
            if include_translation and target_language and target_language != detected_lang:
                translation_result = self.translate_text(text, detected_lang, target_language)
            
            return {
                "original_text": text,
                "detected_language": detection_result,
                "processing": processing_result,
                "translation": translation_result if translation_result else None,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Multilingual text processing failed: {e}")
            return {
                "original_text": text,
                "status": "error",
                "error": str(e)
            }
    
    def _process_text_by_language(self, text: str, language: str) -> Dict[str, Any]:
        """Process text using language-specific models."""
        if not self._spacy_available:
            return {
                "method": "basic",
                "tokens": text.split(),
                "sentences": text.split('.'),
                "note": "spaCy not available - using basic processing"
            }
        
        try:
            import spacy
            
            # Map language codes to spaCy model names
            spacy_models = {
                'en': 'en_core_web_sm',
                'es': 'es_core_news_sm',
                'fr': 'fr_core_news_sm',
                'de': 'de_core_news_sm',
                'it': 'it_core_news_sm',
                'pt': 'pt_core_news_sm',
                'zh': 'zh_core_web_sm',
                'ja': 'ja_core_news_sm',
                'ru': 'ru_core_news_sm'
            }
            
            model_name = spacy_models.get(language, 'en_core_web_sm')
            
            # Load model (with caching)
            if model_name not in self._spacy_models:
                try:
                    self._spacy_models[model_name] = spacy.load(model_name)
                except OSError:
                    # Fallback to English model
                    self.logger.warning(f"Model {model_name} not found, using English model")
                    if 'en_core_web_sm' not in self._spacy_models:
                        try:
                            self._spacy_models['en_core_web_sm'] = spacy.load('en_core_web_sm')
                        except OSError:
                            # No spaCy models available
                            return {
                                "method": "basic",
                                "tokens": text.split(),
                                "sentences": text.split('.'),
                                "note": "No spaCy models available"
                            }
                    model_name = 'en_core_web_sm'
            
            nlp = self._spacy_models[model_name]
            doc = nlp(text)
            
            return {
                "method": "spacy",
                "model": model_name,
                "tokens": [token.text for token in doc],
                "lemmas": [token.lemma_ for token in doc],
                "pos_tags": [token.pos_ for token in doc],
                "entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents],
                "sentences": [sent.text for sent in doc.sents]
            }
            
        except Exception as e:
            self.logger.error(f"spaCy processing failed: {e}")
            return {
                "method": "basic",
                "tokens": text.split(),
                "sentences": text.split('.'),
                "error": str(e)
            }
    
    def translate_text(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str,
        method: str = "auto"
    ) -> Dict[str, Any]:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            method: Translation method ('auto', 'googletrans')
            
        Returns:
            Dictionary containing translation results
        """
        if not text or not text.strip():
            return {
                "translated_text": "",
                "source_language": source_lang,
                "target_language": target_lang,
                "method": method,
                "confidence": 0.0,
                "error": "Empty text"
            }
        
        try:
            if method == "auto" or method == "googletrans":
                if self._googletrans_available:
                    return self._translate_with_googletrans(text, source_lang, target_lang)
            
            # Fallback: return original text with note
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "method": "none",
                "confidence": 0.0,
                "note": "No translation service available"
            }
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "method": method,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _translate_with_googletrans(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> Dict[str, Any]:
        """Translate text using Google Translate."""
        try:
            from googletrans import Translator
            
            translator = Translator()
            result = translator.translate(text, src=source_lang, dest=target_lang)
            
            return {
                "translated_text": result.text,
                "source_language": result.src,
                "target_language": target_lang,
                "method": "googletrans",
                "confidence": 0.9,  # Google Translate is generally reliable
                "detected_source": result.src
            }
            
        except Exception as e:
            self.logger.error(f"Google Translate failed: {e}")
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "method": "googletrans",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Get list of supported languages for each component."""
        supported = {
            "detection": [],
            "processing": [],
            "translation": []
        }
        
        # Language detection support
        if self._langdetect_available:
            supported["detection"].extend([
                "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", 
                "ar", "hi", "nl", "sv", "da", "no", "fi", "pl", "tr", "he"
            ])
        
        # Text processing support (spaCy models)
        if self._spacy_available:
            supported["processing"].extend([
                "en", "es", "fr", "de", "it", "pt", "zh", "ja", "ru", "nl"
            ])
        
        # Translation support
        if self._googletrans_available:
            supported["translation"].extend([
                "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
                "ar", "hi", "nl", "sv", "da", "no", "fi", "pl", "tr", "he",
                "th", "vi", "id", "ms", "tl", "sw", "am", "bn", "gu", "kn",
                "ml", "mr", "ne", "or", "pa", "si", "ta", "te", "ur"
            ])
        
        return supported
    
    def is_available(self) -> bool:
        """Check if multilingual processing is available."""
        return any([
            self._langdetect_available,
            self._spacy_available,
            self._polyglot_available,
            self._googletrans_available
        ])
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about available multilingual components."""
        return {
            "multilingual_available": self.is_available(),
            "langdetect_available": self._langdetect_available,
            "spacy_available": self._spacy_available,
            "polyglot_available": self._polyglot_available,
            "googletrans_available": self._googletrans_available,
            "loaded_spacy_models": list(self._spacy_models.keys()),
            "supported_languages": self.get_supported_languages()
        }


# Convenience functions for easy usage
def detect_document_language(text: str) -> Dict[str, Any]:
    """
    Convenience function to detect document language.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language detection result dictionary
    """
    processor = MultilingualProcessor()
    return processor.detect_language(text)


def process_multilingual_document(
    text: str,
    target_language: Optional[str] = None,
    include_translation: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to process multilingual document.
    
    Args:
        text: Text to process
        target_language: Target language for translation
        include_translation: Whether to include translation
        
    Returns:
        Multilingual processing result dictionary
    """
    processor = MultilingualProcessor()
    return processor.process_multilingual_text(text, target_language, include_translation)
