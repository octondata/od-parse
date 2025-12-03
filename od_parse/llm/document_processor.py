"""
LLM-Powered Document Processor.

Advanced document parsing using state-of-the-art language models
for complex PDF understanding and structured data extraction.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from od_parse.config.llm_config import DocumentComplexity, LLMProvider, get_llm_config
from od_parse.intelligence import DocumentClassifier, DocumentType
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMDocumentProcessor:
    """
    LLM-powered document processor for complex PDF understanding.

    This processor uses advanced language models to extract structured
    information from complex documents with high accuracy.
    """

    def __init__(
        self, model_id: Optional[str] = None, custom_config: Optional[Dict] = None
    ):
        """
        Initialize LLM document processor.

        Args:
            model_id: Specific model to use (optional, will auto-select if None)
            custom_config: Custom configuration overrides
        """
        self.logger = get_logger(__name__)
        self.llm_config = get_llm_config()
        self.model_id = model_id
        self.custom_config = custom_config or {}

        # Initialize document classifier for intelligent routing
        self.classifier = DocumentClassifier()

        # Validate LLM availability
        self._validate_llm_setup()

    def _validate_llm_setup(self) -> None:
        """Validate that LLM is properly configured."""
        available_models = self.llm_config.get_available_models()

        if not available_models:
            raise ValueError(
                "No LLM API keys found. od-parse requires LLM access for document parsing.\n"
                "Please set one of the following environment variables:\n"
                "  OPENAI_API_KEY for OpenAI models\n"
                "  ANTHROPIC_API_KEY for Claude models\n"
                "  GOOGLE_API_KEY for Gemini models\n"
                "  AZURE_OPENAI_API_KEY for Azure OpenAI\n"
                "See README.md for detailed setup instructions."
            )

        if self.model_id and self.model_id not in available_models:
            self.logger.warning(
                f"Requested model {self.model_id} not available. Using default."
            )
            self.model_id = None

        if not self.model_id:
            self.model_id = self.llm_config.default_provider

        self.logger.info(f"Using LLM model: {self.model_id}")

    def process_document(
        self,
        parsed_data: Dict[str, Any],
        document_images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, Any]:
        """
        Process document using LLM for advanced understanding.

        Args:
            parsed_data: Basic parsed data from PDF
            document_images: List of document page images for vision models

        Returns:
            Enhanced document data with LLM analysis
        """
        try:
            # Step 1: Classify document type for intelligent processing
            classification = self.classifier.classify_document(parsed_data)
            doc_type = classification.document_type.value

            self.logger.info(f"Processing {doc_type} document with LLM")

            # Step 2: Select optimal model for document type
            optimal_model = self.llm_config.get_recommended_model(doc_type)
            if optimal_model and optimal_model != self.model_id:
                self.logger.info(
                    f"Switching to optimal model for {doc_type}: {optimal_model}"
                )
                self.model_id = optimal_model

            # Step 3: Get document-specific system prompt
            system_prompt = self.llm_config.get_system_prompt(doc_type)

            # Step 4: Process with LLM
            llm_result = self._process_with_llm(
                parsed_data=parsed_data,
                document_images=document_images,
                system_prompt=system_prompt,
                doc_type=doc_type,
            )

            # Step 5: Combine results
            enhanced_data = {
                **parsed_data,
                "llm_analysis": llm_result,
                "document_classification": {
                    "document_type": classification.document_type.value,
                    "confidence": classification.confidence,
                    "detected_patterns": classification.detected_patterns,
                    "key_indicators": classification.key_indicators,
                    "suggestions": classification.suggestions,
                },
                "processing_metadata": {
                    "model_used": self.model_id,
                    "document_type": doc_type,
                    "processing_strategy": "llm_enhanced",
                    "vision_enabled": bool(document_images),
                },
            }

            return enhanced_data

        except Exception as e:
            self.logger.error(f"LLM document processing failed: {e}")
            # Return original data with error info
            return {
                **parsed_data,
                "llm_analysis": {"error": str(e)},
                "processing_metadata": {
                    "model_used": self.model_id,
                    "processing_strategy": "fallback",
                    "error": str(e),
                },
            }

    def _process_with_llm(
        self,
        parsed_data: Dict[str, Any],
        document_images: Optional[List[Image.Image]],
        system_prompt: str,
        doc_type: str,
    ) -> Dict[str, Any]:
        """Process document with specific LLM."""
        model_config = self.llm_config.models[self.model_id]

        if model_config.provider == LLMProvider.OPENAI:
            return self._process_with_openai(
                parsed_data, document_images, system_prompt, doc_type
            )
        elif model_config.provider == LLMProvider.ANTHROPIC:
            return self._process_with_anthropic(
                parsed_data, document_images, system_prompt, doc_type
            )
        elif model_config.provider == LLMProvider.GOOGLE:
            return self._process_with_google(
                parsed_data, document_images, system_prompt, doc_type
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {model_config.provider}")

    def _process_with_openai(
        self,
        parsed_data: Dict[str, Any],
        document_images: Optional[List[Image.Image]],
        system_prompt: str,
        doc_type: str,
    ) -> Dict[str, Any]:
        """Process with OpenAI models."""
        try:
            import openai

            model_config = self.llm_config.models[self.model_id]
            client = openai.OpenAI(api_key=os.getenv(model_config.api_key_env))

            # Prepare messages
            messages = [{"role": "system", "content": system_prompt}]

            # Add text content
            text_content = parsed_data.get("text", "")
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please analyze this {doc_type} document and extract structured information:\n\n{text_content[:10000]}",  # Limit text length
                    }
                ],
            }

            # Add images if available and model supports vision
            if document_images and model_config.supports_vision:
                for i, image in enumerate(
                    document_images[:3]
                ):  # Limit to first 3 pages
                    # Convert image to base64
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    image_base64 = base64.b64encode(buffer.getvalue()).decode()

                    user_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high",
                            },
                        }
                    )

            messages.append(user_message)

            # Make API call
            response = client.chat.completions.create(
                model=model_config.model_name,
                messages=messages,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                response_format={"type": "json_object"},
            )

            # Parse response
            result = json.loads(response.choices[0].message.content)

            return {
                "extracted_data": result,
                "model_info": {
                    "provider": "openai",
                    "model": model_config.model_name,
                    "tokens_used": response.usage.total_tokens,
                    "cost_estimate": response.usage.total_tokens
                    * model_config.cost_per_1k_tokens
                    / 1000,
                },
                "processing_success": True,
            }

        except Exception as e:
            self.logger.error(f"OpenAI processing failed: {e}")
            return {"extracted_data": {}, "error": str(e), "processing_success": False}

    def _process_with_anthropic(
        self,
        parsed_data: Dict[str, Any],
        document_images: Optional[List[Image.Image]],
        system_prompt: str,
        doc_type: str,
    ) -> Dict[str, Any]:
        """Process with Anthropic Claude models."""
        try:
            import anthropic

            model_config = self.llm_config.models[self.model_id]
            client = anthropic.Anthropic(api_key=os.getenv(model_config.api_key_env))

            # Prepare content
            content = []

            # Add text content
            text_content = parsed_data.get("text", "")
            content.append(
                {
                    "type": "text",
                    "text": f"Please analyze this {doc_type} document and extract structured information in JSON format:\n\n{text_content[:10000]}",
                }
            )

            # Add images if available and model supports vision
            if document_images and model_config.supports_vision:
                for i, image in enumerate(
                    document_images[:3]
                ):  # Limit to first 3 pages
                    # Convert image to base64
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    image_base64 = base64.b64encode(buffer.getvalue()).decode()

                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64,
                            },
                        }
                    )

            # Make API call
            response = client.messages.create(
                model=model_config.model_name,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            )

            # Parse response
            response_text = response.content[0].text
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If not valid JSON, wrap in structure
                result = {"extracted_text": response_text}

            return {
                "extracted_data": result,
                "model_info": {
                    "provider": "anthropic",
                    "model": model_config.model_name,
                    "tokens_used": response.usage.input_tokens
                    + response.usage.output_tokens,
                    "cost_estimate": (
                        response.usage.input_tokens + response.usage.output_tokens
                    )
                    * model_config.cost_per_1k_tokens
                    / 1000,
                },
                "processing_success": True,
            }

        except Exception as e:
            self.logger.error(f"Anthropic processing failed: {e}")
            return {"extracted_data": {}, "error": str(e), "processing_success": False}

    def _process_with_google(
        self,
        parsed_data: Dict[str, Any],
        document_images: Optional[List[Image.Image]],
        system_prompt: str,
        doc_type: str,
    ) -> Dict[str, Any]:
        """Process with Google Gemini models."""
        try:
            import google.generativeai as genai
            import os

            model_config = self.llm_config.models[self.model_id]
            genai.configure(api_key=os.getenv(model_config.api_key_env))

            model = genai.GenerativeModel(model_config.model_name)

            # Prepare content
            content = [
                f"{system_prompt}\n\nPlease analyze this {doc_type} document and extract structured information in JSON format:\n\n{parsed_data.get('text', '')[:10000]}"
            ]

            # Add images if available and model supports vision
            if document_images and model_config.supports_vision:
                content.extend(document_images[:3])  # Add first 3 pages

            # Make API call
            response = model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=model_config.max_tokens,
                    temperature=model_config.temperature,
                ),
            )

            # Parse response
            response_text = response.text
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If not valid JSON, wrap in structure
                result = {"extracted_text": response_text}

            return {
                "extracted_data": result,
                "model_info": {
                    "provider": "google",
                    "model": model_config.model_name,
                    "tokens_used": (
                        response.usage_metadata.total_token_count
                        if hasattr(response, "usage_metadata")
                        else 0
                    ),
                    "cost_estimate": 0,  # Calculate based on actual usage
                },
                "processing_success": True,
            }

        except Exception as e:
            self.logger.error(f"Google processing failed: {e}")
            return {"extracted_data": {}, "error": str(e), "processing_success": False}
