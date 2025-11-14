
"""
This module contains the DocumentSegmenter class, which is responsible for
detecting boundaries between different logical documents within a single PDF file.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import pdfplumber
import pdf2image


class DocumentSegmenter:
    """
    Analyzes a PDF to find boundaries between distinct documents.
    
    Uses a hybrid approach:
    1. Fast, lightweight fingerprinting of pages using traditional libraries.
    2. An LLM tie-breaker for ambiguous cases.
    """

    SIMILARITY_WEIGHTS = {
        "text_length": 0.3,
        "alpha_ratio": 0.1,
        "visual_hash": 0.6,
    }

    def __init__(self, llm_client: Any = None, similarity_threshold: float = 0.9):
        """
        Initializes the DocumentSegmenter.

        Args:
            llm_client: An optional client for a multimodal LLM.
            similarity_threshold: The score above which pages are considered similar.
        """
        self.llm_client = llm_client
        self.similarity_threshold = similarity_threshold

    def segment(self, file_path: str) -> List[List[int]]:
        """
        Segments the PDF into chunks of pages, each representing a distinct document.

        Args:
            file_path: The path to the PDF file.

        Returns:
            A list of page groups, e.g., [[1, 2], [3, 4, 5], [6]].
        """
        try:
            images = pdf2image.convert_from_path(file_path)
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            # If we can't even open the PDF, assume it's a single chunk.
            with pdfplumber.open(file_path) as pdf:
                return [list(range(1, len(pdf.pages) + 1))]

        fingerprints = []
        with pdfplumber.open(file_path) as pdf:
            if len(images) != len(pdf.pages):
                raise ValueError("Page count mismatch between pdf2image and pdfplumber.")

            for i, page in enumerate(pdf.pages):
                # pdfplumber text, pdf2image image
                text = page.extract_text() or ""
                image = np.array(images[i])
                fp = self._get_page_fingerprint(image, text)
                fingerprints.append(fp)
        
        if not fingerprints:
            return []

        # The first page always starts a new document.
        boundaries = [0]
        for i in range(len(fingerprints) - 1):
            fp1 = fingerprints[i]
            fp2 = fingerprints[i+1]
            similarity = self._calculate_similarity(fp1, fp2)

            # TODO: Add logic for the LLM tie-breaker here for ambiguous cases.
            if similarity < self.similarity_threshold:
                # The next page (i+1) is the start of a new document.
                boundaries.append(i + 1)

        # Create chunks from boundaries
        chunks = []
        for i in range(len(boundaries)):
            start_page_idx = boundaries[i]
            end_page_idx = boundaries[i+1] if i + 1 < len(boundaries) else len(fingerprints)
            # Page numbers are 1-based, so add 1 to indices
            chunks.append(list(range(start_page_idx + 1, end_page_idx + 1)))

        return chunks

    def _get_page_fingerprint(self, page_image: np.ndarray, page_text: str) -> Dict[str, Any]:
        """
        Generates a "fast fingerprint" of a single page using lightweight, 
        non-LLM methods.

        Args:
            page_image: A NumPy array of the page image (from OpenCV).
            page_text: The raw text extracted from the page.

        Returns:
            A dictionary of features representing the page's structure.
        """
        # 1. Text-based features
        text_length = len(page_text)
        alpha_ratio = self._calculate_alpha_ratio(page_text)

        # 2. Visual features
        # Convert to grayscale for hashing
        gray_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        # pHash is robust to minor changes and scaling
        p_hash = cv2.img_hash.pHash(gray_image)

        return {
            "text_length": text_length,
            "alpha_ratio": alpha_ratio,
            "visual_hash": p_hash,
        }

    def _calculate_similarity(self, fp1: Dict, fp2: Dict) -> float:
        """
        Calculates a weighted similarity score between two page fingerprints.

        Args:
            fp1: The fingerprint of the first page.
            fp2: The fingerprint of the second page.

        Returns:
            A similarity score between 0.0 (completely different) and 1.0 (identical).
        """
        scores = {}

        # Text length similarity
        len1, len2 = fp1["text_length"], fp2["text_length"]
        if max(len1, len2) > 0:
            scores["text_length"] = min(len1, len2) / max(len1, len2)
        else:
            scores["text_length"] = 1.0

        # Alpha ratio similarity
        ar1, ar2 = fp1["alpha_ratio"], fp2["alpha_ratio"]
        if max(ar1, ar2) > 0:
            scores["alpha_ratio"] = min(ar1, ar2) / max(ar1, ar2)
        else:
            scores["alpha_ratio"] = 1.0

        # Visual hash similarity (based on Hamming distance)
        hash1, hash2 = fp1["visual_hash"], fp2["visual_hash"]
        # The compare method returns the Hamming distance
        distance = hash1.compare(hash2)
        # Normalize to a 0-1 similarity score. pHash has 64 bits.
        scores["visual_hash"] = 1.0 - (distance / 64.0)

        # Calculate weighted average
        total_score = sum(scores[key] * self.SIMILARITY_WEIGHTS[key] for key in scores)
        
        return total_score

    def _is_boundary_with_llm(self, page_image1: Any, page_image2: Any) -> bool:
        """
        Uses a multimodal LLM to determine if there is a document boundary 
        between two pages.

        Args:
            page_image1: The image of the first page.
            page_image2: The image of the second page.

        Returns:
            True if the LLM determines a boundary exists, False otherwise.
        """
        if not self.llm_client:
            # Fallback if no LLM is available.
            # In a real scenario, we might want to be more conservative and
            # assume a boundary in ambiguous cases.
            return False
        pass
