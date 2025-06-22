"""
LLaVA-NeXT Integration

This module provides LLaVA-NeXT integration for advanced document understanding
using vision-language models, with fallback to traditional extraction methods.
"""

import os
import warnings
from typing import Union, Optional, Dict, Any, List
from pathlib import Path

import numpy as np
from PIL import Image

from od_parse.config import get_advanced_config
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLaVANextEngine:
    """
    LLaVA-NeXT-based document understanding engine.
    
    This class provides advanced document understanding using LLaVA-NeXT
    vision-language models, with automatic fallback to traditional methods
    when LLaVA-NeXT dependencies are not available.
    """
    
    def __init__(self, 
                 model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
                 device: str = "auto",
                 load_in_4bit: bool = True):
        """
        Initialize the LLaVA-NeXT engine.
        
        Args:
            model_name: LLaVA-NeXT model to use. Options:
                - llava-hf/llava-v1.6-mistral-7b-hf (default)
                - llava-hf/llava-v1.6-vicuna-7b-hf
                - llava-hf/llava-v1.6-vicuna-13b-hf
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            load_in_4bit: Whether to load model in 4-bit quantization
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        
        # Model components
        self.processor = None
        self.model = None
        
        self._is_available = False
        self._fallback_engine = None
        
        # Initialize LLaVA-NeXT
        self._initialize_llava_next()
        
        # Initialize fallback engine if LLaVA-NeXT is not available
        if not self._is_available:
            self._initialize_fallback()
    
    def _initialize_llava_next(self) -> bool:
        """Initialize LLaVA-NeXT model and processor."""
        config = get_advanced_config()
        
        if not config.is_feature_enabled("llava_next"):
            self.logger.info("LLaVA-NeXT feature is disabled. Use config.enable_feature('llava_next') to enable.")
            return False
        
        if not config.is_feature_available("llava_next"):
            self.logger.warning("LLaVA-NeXT dependencies not available. Install with: pip install od-parse[llava_next]")
            return False
        
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            import torch
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Loading LLaVA-NeXT model: {self.model_name} on {self.device}")
            
            # Load processor
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
            
            # Load model with optional quantization
            if self.load_in_4bit and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.model = LlavaNextForConditionalGeneration.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    
                    self.logger.info("Model loaded with 4-bit quantization")
                    
                except ImportError:
                    self.logger.warning("bitsandbytes not available, loading model without quantization")
                    self.model = LlavaNextForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16
                    )
                    self.model.to(self.device)
            else:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.model.to(self.device)
            
            self.model.eval()
            
            self._is_available = True
            self.logger.info("LLaVA-NeXT initialized successfully")
            return True
            
        except ImportError as e:
            self.logger.warning(f"LLaVA-NeXT dependencies not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize LLaVA-NeXT: {e}")
            return False
    
    def _initialize_fallback(self):
        """Initialize fallback document understanding engine."""
        self.logger.info("Using fallback document understanding (basic text extraction)")
        self._fallback_engine = "basic"
    
    def understand_document(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: str = "Describe this document in detail, including its structure, content, and any important information.",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Understand document content using LLaVA-NeXT or fallback engine.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            prompt: Prompt for document understanding
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary containing document understanding results
        """
        try:
            # Convert input to PIL Image
            pil_image = self._prepare_image(image)
            
            if self._is_available:
                return self._understand_with_llava_next(
                    pil_image, prompt, max_new_tokens, temperature, **kwargs
                )
            else:
                return self._understand_with_fallback(pil_image, prompt, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Error understanding document: {e}")
            return {
                "understanding": "",
                "confidence": 0.0,
                "engine": "error",
                "error": str(e)
            }
    
    def _prepare_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _understand_with_llava_next(
        self, 
        image: Image.Image, 
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Understand document using LLaVA-NeXT."""
        try:
            import torch
            
            # Prepare conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            prompt_text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=prompt_text,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode response
            response = self.processor.batch_decode(
                generate_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Estimate confidence based on response quality
            confidence = self._estimate_response_confidence(response)
            
            return {
                "understanding": response.strip(),
                "confidence": confidence,
                "engine": "llava_next",
                "model": self.model_name,
                "device": self.device,
                "prompt": prompt
            }
            
        except Exception as e:
            self.logger.error(f"LLaVA-NeXT understanding failed: {e}")
            # Fallback to traditional understanding
            return self._understand_with_fallback(image, prompt)
    
    def _understand_with_fallback(
        self, 
        image: Image.Image, 
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Understand document using fallback engine."""
        try:
            # Basic fallback: describe image properties
            width, height = image.size
            mode = image.mode
            
            # Simple analysis
            understanding = f"""This appears to be a document image with the following properties:
- Dimensions: {width} x {height} pixels
- Color mode: {mode}
- The image contains what appears to be document content that would require advanced vision-language models for detailed understanding.

To get detailed document understanding, please enable the LLaVA-NeXT feature and install the required dependencies:
pip install od-parse[llava_next]

Then enable the feature:
from od_parse.config import get_advanced_config
config = get_advanced_config()
config.enable_feature('llava_next')
"""
            
            return {
                "understanding": understanding,
                "confidence": 0.3,  # Low confidence for basic analysis
                "engine": "fallback",
                "note": "Basic image analysis - install LLaVA-NeXT for advanced understanding"
            }
            
        except Exception as e:
            self.logger.error(f"Fallback understanding failed: {e}")
            return {
                "understanding": "Unable to analyze document",
                "confidence": 0.0,
                "engine": "fallback",
                "error": str(e)
            }
    
    def _estimate_response_confidence(self, response: str) -> float:
        """
        Estimate confidence score for LLaVA-NeXT response.
        
        This is a simplified approach based on response characteristics.
        """
        if not response or not response.strip():
            return 0.0
        
        # Base confidence for LLaVA-NeXT
        score = 0.8
        
        # Adjust based on response characteristics
        response_length = len(response.strip())
        
        if response_length < 20:
            score *= 0.6  # Very short responses are less reliable
        elif response_length > 200:
            score *= 1.1  # Longer, detailed responses are more reliable
        
        # Check for uncertainty indicators
        uncertainty_phrases = [
            "i'm not sure", "unclear", "difficult to determine",
            "cannot see clearly", "appears to be", "seems like"
        ]
        
        uncertainty_count = sum(1 for phrase in uncertainty_phrases 
                              if phrase in response.lower())
        
        if uncertainty_count > 0:
            score *= (0.8 ** uncertainty_count)  # Reduce confidence for uncertainty
        
        # Check for specific document elements mentioned
        document_elements = [
            "table", "form", "header", "footer", "paragraph",
            "list", "bullet", "number", "date", "signature"
        ]
        
        element_count = sum(1 for element in document_elements 
                          if element in response.lower())
        
        if element_count > 2:
            score *= 1.1  # Boost confidence for detailed structural analysis
        
        return min(score, 1.0)
    
    def extract_structured_information(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        information_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract structured information from document using targeted prompts.
        
        Args:
            image: Input image
            information_type: Type of information to extract
                - "general": General document understanding
                - "tables": Focus on table extraction
                - "forms": Focus on form field extraction
                - "metadata": Focus on document metadata
                - "summary": Generate document summary
            **kwargs: Additional arguments
            
        Returns:
            Structured information extraction results
        """
        # Define prompts for different information types
        prompts = {
            "general": "Analyze this document and provide a detailed description of its content, structure, and purpose.",
            "tables": "Identify and describe all tables in this document. For each table, describe its structure, headers, and key data.",
            "forms": "Identify all form fields in this document. List the field names, types, and any filled values you can see.",
            "metadata": "Extract metadata from this document including title, author, date, document type, and any other identifying information.",
            "summary": "Provide a concise summary of this document's main points and key information."
        }
        
        prompt = prompts.get(information_type, prompts["general"])
        
        # Add specific instructions based on information type
        if information_type == "tables":
            prompt += " Format your response to clearly separate each table and its contents."
        elif information_type == "forms":
            prompt += " Present the form fields in a structured format."
        elif information_type == "metadata":
            prompt += " Present the metadata in a structured format with clear labels."
        
        result = self.understand_document(image, prompt, **kwargs)
        result["information_type"] = information_type
        
        return result
    
    def batch_understand_documents(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        prompt: str = "Describe this document in detail.",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Understand multiple documents in batch.
        
        Args:
            images: List of images to process
            prompt: Prompt for document understanding
            **kwargs: Additional arguments
            
        Returns:
            List of understanding results
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.understand_document(image, prompt, **kwargs)
                result["image_index"] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process image {i}: {e}")
                results.append({
                    "understanding": "",
                    "confidence": 0.0,
                    "engine": "error",
                    "image_index": i,
                    "error": str(e)
                })
        
        return results
    
    def is_available(self) -> bool:
        """Check if LLaVA-NeXT is available."""
        return self._is_available
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the current document understanding engine."""
        return {
            "llava_next_available": self._is_available,
            "model_name": self.model_name if self._is_available else None,
            "device": self.device if self._is_available else None,
            "load_in_4bit": self.load_in_4bit if self._is_available else None,
            "fallback_available": self._fallback_engine is not None,
            "current_engine": "llava_next" if self._is_available else "fallback"
        }


# Convenience function for easy usage
def understand_document_with_llava(
    image: Union[str, Path, Image.Image, np.ndarray],
    prompt: str = "Describe this document in detail.",
    model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to understand document using LLaVA-NeXT.
    
    Args:
        image: Input image
        prompt: Prompt for understanding
        model_name: LLaVA-NeXT model to use
        **kwargs: Additional arguments
        
    Returns:
        Document understanding result dictionary
    """
    engine = LLaVANextEngine(model_name=model_name)
    return engine.understand_document(image, prompt, **kwargs)
