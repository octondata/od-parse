"""
Vision Language Model (VLM) Processor for enhanced document understanding.

This module integrates state-of-the-art Vision Language Models like Qwen 2.5 VL
to improve document parsing by leveraging both visual and textual informations.
"""

import os
import logging
import base64
import json
from io import BytesIO
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import requests
from PIL import Image
import numpy as np

from od_parse.utils.logging_utils import get_logger
from od_parse.config.settings import get_config, load_config


class VLMProcessor:
    """
    Processor for enhanced document understanding.
    
    This class provides methods for using VLMs to analyze document images,
    extract information, and enhance the parsing results with visual context.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the VLM processor with configuration options.
        
        Args:
            config: Configuration dictionary with the following options:
                - model: The VLM model to use (default: "qwen2.5-vl")
                - api_key: API key for the model service
                - api_base: Base URL for the API
                - max_tokens: Maximum tokens for generation (default: 1024)
                - temperature: Temperature for generation (default: 0.2)
                - system_prompt: System prompt for the model (default: get_config("system_prompts.document_analysis"))
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Load configuration if not already loaded
        if not get_config():
            load_config()
            
        # Set default configuration values
        model_type = self.config.get("model_type", "qwen")
        self.model = self.config.get("model", get_config(f"vlm_models.{model_type}"))
        
        # Get API provider based on model type
        api_provider = "openai"  # Default provider for Qwen
        if model_type == "claude":
            api_provider = "anthropic"
        elif model_type == "gemini":
            api_provider = "google"
            
        # Get API key from config, with fallbacks
        self.api_key = self.config.get(
            "api_key",
            get_config(f"api_keys.{api_provider}") or os.environ.get(f"{api_provider.upper()}_API_KEY")
        )
            
        self.api_base = self.config.get("api_base", get_config(f"api_endpoints.{api_provider}"))
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.temperature = self.config.get("temperature", 0.2)
        
        # Get system prompt from config
        self.system_prompt = self.config.get(
            "system_prompt", 
            get_config("system_prompts.document_analysis")
        )
        
        # Initialize the VLM client
        self._init_vlm_client()
    
    def _init_vlm_client(self):
        """Initialize the VLM client based on configuration."""
        self.client = None
        
        if self.model.startswith("qwen"):
            try:
                from openai import OpenAI
                if not self.api_key:
                    self.logger.warning("API key not provided. VLM processing will not work.")
                    return
                
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
                self.logger.info(f"Initialized VLM model: {self.model}")
            except ImportError:
                self.logger.error("OpenAI package not installed. Please install it with: pip install openai")
        
        elif self.model.startswith("claude"):
            try:
                from anthropic import Anthropic
                if not self.api_key:
                    self.logger.warning("Anthropic API key not provided. VLM processing will not work.")
                    return
                
                self.client = Anthropic(api_key=self.api_key)
                self.logger.info(f"Initialized VLM model: {self.model}")
            except ImportError:
                self.logger.error("Anthropic package not installed. Please install it with: pip install anthropic")
        
        elif self.model.startswith("gemini"):
            try:
                import google.generativeai as genai
                if not self.api_key:
                    self.logger.warning("Google API key not provided. VLM processing will not work.")
                    return
                
                genai.configure(api_key=self.api_key)
                self.client = genai
                self.logger.info(f"Initialized VLM model: {self.model}")
            except ImportError:
                self.logger.error("Google GenerativeAI package not installed. Please install it with: pip install google-generativeai")
        
        else:
            self.logger.error(f"Unsupported VLM model: {self.model}")
    
    def process_document_image(self, image_path: Union[str, Path], prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document image using the VLM.
        
        Args:
            image_path: Path to the document image
            prompt: Custom prompt to guide the VLM analysis
            
        Returns:
            Dictionary containing the VLM analysis results
        """
        if not self.client:
            self.logger.error("VLM client not initialized")
            return {"error": "VLM client not initialized"}
        
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Use the provided prompt or default to a general document analysis prompt
            if not prompt:
                prompt = "Analyze this document and extract all relevant information including text, tables, and form elements."
            
            # Process with the appropriate VLM
            if self.model.startswith("qwen"):
                return self._process_with_qwen(image, prompt)
            elif self.model.startswith("claude"):
                return self._process_with_claude(image, prompt)
            elif self.model.startswith("gemini"):
                return self._process_with_gemini(image, prompt)
            else:
                self.logger.error(f"Unsupported VLM model: {self.model}")
                return {"error": f"Unsupported VLM model: {self.model}"}
        
        except Exception as e:
            self.logger.error(f"Error processing document image: {str(e)}")
            return {"error": str(e)}
    
    def _process_with_qwen(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Process the document image with Qwen VLM."""
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            result = {
                "model": self.model,
                "analysis": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing with Qwen: {str(e)}")
            return {"error": str(e)}
    
    def _process_with_claude(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Process the document image with Claude VLM."""
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}}
                        ]
                    }
                ]
            )
            
            result = {
                "model": self.model,
                "analysis": response.content[0].text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing with Claude: {str(e)}")
            return {"error": str(e)}
    
    def _process_with_gemini(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Process the document image with Gemini VLM."""
        try:
            model = self.client.GenerativeModel(self.model)
            response = model.generate_content(
                [prompt, image],
                generation_config=self.client.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            result = {
                "model": self.model,
                "analysis": response.text
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing with Gemini: {str(e)}")
            return {"error": str(e)}
    
    def enhance_parsing_results(self, parsed_data: Dict[str, Any], image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Enhance parsing results with VLM analysis.
        
        Args:
            parsed_data: The parsed document data from other parsers
            image_path: Path to the document image
            
        Returns:
            Enhanced document data with VLM insights
        """
        # Process the document image with VLM
        vlm_results = self.process_document_image(
            image_path, 
            "Analyze this document and identify any information that might be missed by traditional OCR, "
            "especially handwritten text, complex tables, and form elements."
        )
        
        # If there was an error in VLM processing, return the original data
        if "error" in vlm_results:
            self.logger.warning(f"VLM processing error: {vlm_results['error']}. Using original parsing results.")
            parsed_data["vlm_analysis"] = {"status": "error", "message": vlm_results["error"]}
            return parsed_data
        
        # Add VLM analysis to the parsed data
        parsed_data["vlm_analysis"] = {
            "model": vlm_results["model"],
            "analysis": vlm_results["analysis"]
        }
        
        # Try to extract structured information from the VLM analysis
        try:
            # Ask the VLM to provide structured information
            structured_prompt = (
                "Based on the document image, provide the following structured information in JSON format:\n"
                "1. All tables with their data\n"
                "2. All form fields with their values\n"
                "3. Any handwritten text\n"
                "4. Document title and key metadata\n"
                "Format your response as valid JSON."
            )
            
            structured_results = self.process_document_image(image_path, structured_prompt)
            
            if "error" not in structured_results:
                # Try to extract JSON from the response
                analysis_text = structured_results["analysis"]
                json_start = analysis_text.find("{")
                json_end = analysis_text.rfind("}")
                
                if json_start >= 0 and json_end >= 0:
                    json_str = analysis_text[json_start:json_end+1]
                    try:
                        structured_data = json.loads(json_str)
                        parsed_data["vlm_structured_data"] = structured_data
                    except json.JSONDecodeError:
                        self.logger.warning("Could not parse structured data as JSON")
        
        except Exception as e:
            self.logger.warning(f"Error extracting structured information: {str(e)}")
        
        return parsed_data
    
    def extract_tables_with_vlm(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Extract tables from a document image using VLM.
        
        Args:
            image_path: Path to the document image
            
        Returns:
            List of extracted tables
        """
        prompt = get_config("system_prompts.table_extraction")
        
        vlm_results = self.process_document_image(image_path, prompt)
        
        if "error" in vlm_results:
            self.logger.warning(f"VLM table extraction error: {vlm_results['error']}")
            return []
        
        # Try to extract tables from the VLM response
        tables = []
        try:
            analysis_text = vlm_results["analysis"]
            
            # Look for JSON in the response
            json_start = analysis_text.find("{")
            json_end = analysis_text.rfind("}")
            
            if json_start >= 0 and json_end >= 0:
                json_str = analysis_text[json_start:json_end+1]
                try:
                    table_data = json.loads(json_str)
                    if isinstance(table_data, dict):
                        tables.append(table_data)
                    elif isinstance(table_data, list):
                        tables.extend(table_data)
                except json.JSONDecodeError:
                    self.logger.warning("Could not parse table data as JSON")
        
        except Exception as e:
            self.logger.warning(f"Error extracting tables: {str(e)}")
        
        return tables
    
    def extract_form_fields_with_vlm(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract form fields from a document image using VLM.
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dictionary of form fields and their values
        """
        prompt = get_config("system_prompts.form_extraction")
        
        vlm_results = self.process_document_image(image_path, prompt)
        
        if "error" in vlm_results:
            self.logger.warning(f"VLM form field extraction error: {vlm_results['error']}")
            return {}
        
        # Try to extract form fields from the VLM response
        form_fields = {}
        try:
            analysis_text = vlm_results["analysis"]
            
            # Look for JSON in the response
            json_start = analysis_text.find("{")
            json_end = analysis_text.rfind("}")
            
            if json_start >= 0 and json_end >= 0:
                json_str = analysis_text[json_start:json_end+1]
                try:
                    form_fields = json.loads(json_str)
                except json.JSONDecodeError:
                    self.logger.warning("Could not parse form fields as JSON")
        
        except Exception as e:
            self.logger.warning(f"Error extracting form fields: {str(e)}")
        
        return form_fields
