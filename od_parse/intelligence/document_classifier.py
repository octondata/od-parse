"""
Smart Document Classification

Advanced document type detection and analysis using multiple signals
including text patterns, structure, metadata, and content analysis.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import Counter

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DocumentType(Enum):
    """Document type classification with confidence scoring."""
    
    # Tax Documents
    TAX_FORM_1040 = "tax_form_1040"
    TAX_FORM_1040EZ = "tax_form_1040ez"
    TAX_FORM_1040A = "tax_form_1040a"
    TAX_FORM_W2 = "tax_form_w2"
    TAX_FORM_W4 = "tax_form_w4"
    TAX_FORM_1099 = "tax_form_1099"
    TAX_SCHEDULE_A = "tax_schedule_a"
    TAX_SCHEDULE_B = "tax_schedule_b"
    TAX_SCHEDULE_C = "tax_schedule_c"
    TAX_SCHEDULE_D = "tax_schedule_d"
    
    # Financial Documents
    BANK_STATEMENT = "bank_statement"
    CREDIT_CARD_STATEMENT = "credit_card_statement"
    INVESTMENT_STATEMENT = "investment_statement"
    LOAN_DOCUMENT = "loan_document"
    MORTGAGE_DOCUMENT = "mortgage_document"
    INSURANCE_POLICY = "insurance_policy"
    FINANCIAL_REPORT = "financial_report"
    
    # Business Documents
    INVOICE = "invoice"
    RECEIPT = "receipt"
    PURCHASE_ORDER = "purchase_order"
    CONTRACT = "contract"
    AGREEMENT = "agreement"
    PROPOSAL = "proposal"
    QUOTE = "quote"
    
    # Legal Documents
    LEGAL_CONTRACT = "legal_contract"
    COURT_DOCUMENT = "court_document"
    PATENT = "patent"
    TRADEMARK = "trademark"
    WILL_TESTAMENT = "will_testament"
    POWER_OF_ATTORNEY = "power_of_attorney"
    
    # Medical Documents
    MEDICAL_RECORD = "medical_record"
    PRESCRIPTION = "prescription"
    LAB_REPORT = "lab_report"
    INSURANCE_CLAIM = "insurance_claim"
    MEDICAL_BILL = "medical_bill"
    
    # Academic Documents
    RESEARCH_PAPER = "research_paper"
    THESIS = "thesis"
    ACADEMIC_TRANSCRIPT = "academic_transcript"
    DIPLOMA = "diploma"
    CERTIFICATE = "certificate"
    
    # Government Documents
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    BIRTH_CERTIFICATE = "birth_certificate"
    SOCIAL_SECURITY_CARD = "social_security_card"
    GOVERNMENT_FORM = "government_form"
    
    # General Documents
    RESUME = "resume"
    COVER_LETTER = "cover_letter"
    LETTER = "letter"
    MEMO = "memo"
    REPORT = "report"
    MANUAL = "manual"
    BROCHURE = "brochure"
    CATALOG = "catalog"
    
    # Unknown
    UNKNOWN = "unknown"


@dataclass
class DocumentAnalysis:
    """Complete document analysis results."""
    document_type: DocumentType
    confidence: float
    detected_patterns: List[str]
    key_indicators: Dict[str, Any]
    metadata: Dict[str, Any]
    suggestions: List[str]


class DocumentClassifier:
    """
    Smart document classifier using multiple analysis techniques:
    - Text pattern matching
    - Structural analysis
    - Keyword frequency analysis
    - Format detection
    - Content analysis
    """
    
    def __init__(self):
        """Initialize the document classifier."""
        self.logger = get_logger(__name__)
        self._load_classification_patterns()
    
    def classify_document(self, parsed_data: Dict[str, Any]) -> DocumentAnalysis:
        """
        Classify a document using multiple intelligence signals.
        
        Args:
            parsed_data: Raw parsed data from PDF
            
        Returns:
            DocumentAnalysis with classification results
        """
        try:
            text = parsed_data.get('text', '')
            tables = parsed_data.get('tables', [])
            forms = parsed_data.get('forms', [])
            metadata = parsed_data.get('metadata', {})
            
            # Multi-signal analysis
            pattern_signals = self._analyze_text_patterns(text)
            structure_signals = self._analyze_document_structure(text, tables, forms)
            keyword_signals = self._analyze_keywords(text)
            format_signals = self._analyze_format_indicators(text, metadata)
            content_signals = self._analyze_content_semantics(text)
            
            # Combine all signals
            all_signals = {
                **pattern_signals,
                **structure_signals, 
                **keyword_signals,
                **format_signals,
                **content_signals
            }
            
            # Determine best classification
            document_type, confidence = self._determine_classification(all_signals)
            
            # Extract key indicators
            key_indicators = self._extract_key_indicators(text, document_type)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(document_type, confidence, all_signals)
            
            return DocumentAnalysis(
                document_type=document_type,
                confidence=confidence,
                detected_patterns=list(all_signals.keys()),
                key_indicators=key_indicators,
                metadata={
                    "text_length": len(text),
                    "table_count": len(tables),
                    "form_count": len(forms),
                    "signal_scores": all_signals
                },
                suggestions=suggestions
            )
            
        except Exception as e:
            self.logger.error(f"Document classification failed: {e}")
            return DocumentAnalysis(
                document_type=DocumentType.UNKNOWN,
                confidence=0.0,
                detected_patterns=[],
                key_indicators={},
                metadata={"error": str(e)},
                suggestions=["Manual review recommended due to classification error"]
            )
    
    def _load_classification_patterns(self):
        """Load classification patterns for different document types."""
        self.patterns = {
            # Tax Documents
            DocumentType.TAX_FORM_1040: [
                r"form\s+1040",
                r"u\.?s\.?\s+individual\s+income\s+tax\s+return",
                r"department\s+of\s+the\s+treasury",
                r"internal\s+revenue\s+service",
                r"adjusted\s+gross\s+income",
                r"taxable\s+income",
                r"filing\s+status"
            ],
            
            DocumentType.TAX_FORM_W2: [
                r"form\s+w-?2",
                r"wage\s+and\s+tax\s+statement",
                r"employer\s+identification\s+number",
                r"wages,?\s+tips,?\s+other\s+compensation",
                r"federal\s+income\s+tax\s+withheld"
            ],
            
            DocumentType.TAX_FORM_1099: [
                r"form\s+1099",
                r"miscellaneous\s+income",
                r"nonemployee\s+compensation",
                r"payer\s+information",
                r"recipient\s+information"
            ],
            
            # Financial Documents
            DocumentType.BANK_STATEMENT: [
                r"bank\s+statement",
                r"account\s+summary",
                r"beginning\s+balance",
                r"ending\s+balance",
                r"deposits?\s+and\s+credits?",
                r"withdrawals?\s+and\s+debits?",
                r"statement\s+period"
            ],
            
            DocumentType.CREDIT_CARD_STATEMENT: [
                r"credit\s+card\s+statement",
                r"account\s+number",
                r"payment\s+due\s+date",
                r"minimum\s+payment",
                r"previous\s+balance",
                r"new\s+balance",
                r"available\s+credit"
            ],
            
            # Business Documents
            DocumentType.INVOICE: [
                r"invoice",
                r"bill\s+to",
                r"invoice\s+number",
                r"due\s+date",
                r"amount\s+due",
                r"subtotal",
                r"total\s+amount"
            ],
            
            DocumentType.RECEIPT: [
                r"receipt",
                r"thank\s+you\s+for\s+your\s+purchase",
                r"transaction\s+id",
                r"items?\s+purchased",
                r"total\s+paid",
                r"change\s+due"
            ],
            
            DocumentType.CONTRACT: [
                r"contract",
                r"agreement",
                r"terms\s+and\s+conditions",
                r"party\s+of\s+the\s+first\s+part",
                r"whereas",
                r"in\s+consideration\s+of",
                r"signature"
            ],
            
            # Legal Documents
            DocumentType.LEGAL_CONTRACT: [
                r"legal\s+contract",
                r"attorney",
                r"law\s+firm",
                r"legal\s+counsel",
                r"jurisdiction",
                r"governing\s+law",
                r"dispute\s+resolution"
            ],
            
            # Medical Documents
            DocumentType.MEDICAL_RECORD: [
                r"medical\s+record",
                r"patient\s+name",
                r"date\s+of\s+birth",
                r"diagnosis",
                r"treatment",
                r"physician",
                r"medical\s+history"
            ],
            
            # Academic Documents
            DocumentType.RESEARCH_PAPER: [
                r"abstract",
                r"introduction",
                r"methodology",
                r"results",
                r"conclusion",
                r"references",
                r"bibliography"
            ],
            
            # Government Documents
            DocumentType.PASSPORT: [
                r"passport",
                r"united\s+states\s+of\s+america",
                r"nationality",
                r"place\s+of\s+birth",
                r"date\s+of\s+issue",
                r"date\s+of\s+expiration"
            ]
        }
    
    def _analyze_text_patterns(self, text: str) -> Dict[str, float]:
        """Analyze text patterns for document classification."""
        text_lower = text.lower()
        pattern_scores = {}
        
        for doc_type, patterns in self.patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches += 1
                    score += 1.0
            
            if matches > 0:
                # Normalize score by pattern count
                normalized_score = score / len(patterns)
                pattern_scores[f"pattern_{doc_type.value}"] = normalized_score
        
        return pattern_scores
    
    def _analyze_document_structure(self, text: str, tables: List, forms: List) -> Dict[str, float]:
        """Analyze document structure for classification clues."""
        structure_scores = {}
        
        # Table analysis
        if len(tables) > 0:
            structure_scores["has_tables"] = min(len(tables) / 10.0, 1.0)
            
            # Analyze table content for financial patterns
            table_text = ""
            for table in tables[:3]:  # Check first 3 tables
                if isinstance(table, dict) and 'data' in table:
                    for row in table['data'][:5]:  # Check first 5 rows
                        table_text += " ".join(str(cell) for cell in row if cell) + " "
            
            if re.search(r'\$[\d,]+\.?\d*', table_text):
                structure_scores["financial_tables"] = 0.8
        
        # Form analysis
        if len(forms) > 0:
            structure_scores["has_forms"] = min(len(forms) / 20.0, 1.0)
            
            # Check for tax form patterns
            form_text = " ".join(str(form) for form in forms[:10])
            if re.search(r'(ssn|social\s+security|ein|tax)', form_text.lower()):
                structure_scores["tax_forms"] = 0.9
        
        # Text structure analysis
        lines = text.split('\n')
        if len(lines) > 50:
            structure_scores["long_document"] = 0.7
        
        # Check for signature blocks
        if re.search(r'signature|signed|date.*signed', text.lower()):
            structure_scores["has_signatures"] = 0.6
        
        return structure_scores
    
    def _analyze_keywords(self, text: str) -> Dict[str, float]:
        """Analyze keyword frequency for document classification."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_freq = Counter(words)
        
        keyword_scores = {}
        
        # Financial keywords
        financial_keywords = ['tax', 'income', 'payment', 'amount', 'total', 'balance', 'account']
        financial_score = sum(word_freq.get(word, 0) for word in financial_keywords)
        if financial_score > 0:
            keyword_scores["financial_keywords"] = min(financial_score / 100.0, 1.0)
        
        # Legal keywords
        legal_keywords = ['contract', 'agreement', 'terms', 'conditions', 'party', 'whereas']
        legal_score = sum(word_freq.get(word, 0) for word in legal_keywords)
        if legal_score > 0:
            keyword_scores["legal_keywords"] = min(legal_score / 50.0, 1.0)
        
        # Medical keywords
        medical_keywords = ['patient', 'diagnosis', 'treatment', 'medical', 'doctor', 'physician']
        medical_score = sum(word_freq.get(word, 0) for word in medical_keywords)
        if medical_score > 0:
            keyword_scores["medical_keywords"] = min(medical_score / 30.0, 1.0)
        
        return keyword_scores
    
    def _analyze_format_indicators(self, text: str, metadata: Dict) -> Dict[str, float]:
        """Analyze format indicators for classification."""
        format_scores = {}
        
        # Check for common format patterns
        if re.search(r'\d{3}-\d{2}-\d{4}', text):  # SSN pattern
            format_scores["has_ssn"] = 0.9
        
        if re.search(r'\d{2}-\d{7}', text):  # EIN pattern
            format_scores["has_ein"] = 0.8
        
        if re.search(r'\$[\d,]+\.?\d*', text):  # Currency pattern
            format_scores["has_currency"] = 0.7
        
        if re.search(r'\d{1,2}/\d{1,2}/\d{4}', text):  # Date pattern
            format_scores["has_dates"] = 0.5
        
        # Metadata analysis
        if metadata:
            title = metadata.get('title', '').lower()
            if 'tax' in title or '1040' in title:
                format_scores["tax_in_title"] = 0.9
            elif 'invoice' in title:
                format_scores["invoice_in_title"] = 0.9
            elif 'statement' in title:
                format_scores["statement_in_title"] = 0.8
        
        return format_scores
    
    def _analyze_content_semantics(self, text: str) -> Dict[str, float]:
        """Analyze content semantics for advanced classification."""
        semantic_scores = {}
        text_lower = text.lower()
        
        # Tax document semantics
        tax_phrases = [
            'adjusted gross income', 'taxable income', 'filing status',
            'standard deduction', 'itemized deduction', 'tax liability'
        ]
        tax_matches = sum(1 for phrase in tax_phrases if phrase in text_lower)
        if tax_matches > 0:
            semantic_scores["tax_semantics"] = min(tax_matches / len(tax_phrases), 1.0)
        
        # Financial semantics
        financial_phrases = [
            'account balance', 'transaction history', 'interest rate',
            'payment due', 'credit limit', 'available balance'
        ]
        financial_matches = sum(1 for phrase in financial_phrases if phrase in text_lower)
        if financial_matches > 0:
            semantic_scores["financial_semantics"] = min(financial_matches / len(financial_phrases), 1.0)
        
        # Legal semantics
        legal_phrases = [
            'terms and conditions', 'party of the first part', 'in consideration of',
            'governing law', 'dispute resolution', 'force majeure'
        ]
        legal_matches = sum(1 for phrase in legal_phrases if phrase in text_lower)
        if legal_matches > 0:
            semantic_scores["legal_semantics"] = min(legal_matches / len(legal_phrases), 1.0)
        
        return semantic_scores
    
    def _determine_classification(self, signals: Dict[str, float]) -> Tuple[DocumentType, float]:
        """Determine the best document classification from all signals."""
        if not signals:
            return DocumentType.UNKNOWN, 0.0
        
        # Aggregate scores by document type
        type_scores = {}
        
        for signal_name, score in signals.items():
            # Extract document type from signal name
            if signal_name.startswith('pattern_'):
                doc_type_str = signal_name.replace('pattern_', '')
                try:
                    doc_type = DocumentType(doc_type_str)
                    type_scores[doc_type] = type_scores.get(doc_type, 0.0) + score * 2.0  # Pattern matches are strong signals
                except ValueError:
                    continue
            
            # Handle semantic and structural signals
            elif 'tax' in signal_name:
                type_scores[DocumentType.TAX_FORM_1040] = type_scores.get(DocumentType.TAX_FORM_1040, 0.0) + score
            elif 'financial' in signal_name:
                type_scores[DocumentType.BANK_STATEMENT] = type_scores.get(DocumentType.BANK_STATEMENT, 0.0) + score
            elif 'legal' in signal_name:
                type_scores[DocumentType.LEGAL_CONTRACT] = type_scores.get(DocumentType.LEGAL_CONTRACT, 0.0) + score
            elif 'medical' in signal_name:
                type_scores[DocumentType.MEDICAL_RECORD] = type_scores.get(DocumentType.MEDICAL_RECORD, 0.0) + score
        
        if not type_scores:
            return DocumentType.UNKNOWN, 0.0
        
        # Find best match
        best_type = max(type_scores.items(), key=lambda x: x[1])
        confidence = min(best_type[1] / 3.0, 1.0)  # Normalize confidence
        
        return best_type[0], confidence
    
    def _extract_key_indicators(self, text: str, doc_type: DocumentType) -> Dict[str, Any]:
        """Extract key indicators based on document type."""
        indicators = {}
        
        if doc_type in [DocumentType.TAX_FORM_1040, DocumentType.TAX_FORM_W2]:
            # Extract tax-related indicators
            ssn_match = re.search(r'(\d{3}-\d{2}-\d{4})', text)
            if ssn_match:
                indicators["ssn_found"] = ssn_match.group(1)
            
            ein_match = re.search(r'(\d{2}-\d{7})', text)
            if ein_match:
                indicators["ein_found"] = ein_match.group(1)
        
        elif doc_type in [DocumentType.BANK_STATEMENT, DocumentType.CREDIT_CARD_STATEMENT]:
            # Extract financial indicators
            account_match = re.search(r'account\s+(?:number|#)?\s*:?\s*(\d+)', text.lower())
            if account_match:
                indicators["account_number"] = account_match.group(1)
        
        elif doc_type == DocumentType.INVOICE:
            # Extract invoice indicators
            invoice_match = re.search(r'invoice\s+(?:number|#)?\s*:?\s*([A-Za-z0-9-]+)', text.lower())
            if invoice_match:
                indicators["invoice_number"] = invoice_match.group(1)
        
        return indicators
    
    def _generate_suggestions(self, doc_type: DocumentType, confidence: float, signals: Dict[str, float]) -> List[str]:
        """Generate processing suggestions based on classification."""
        suggestions = []
        
        if confidence < 0.5:
            suggestions.append("Low confidence classification - manual review recommended")
        
        if doc_type == DocumentType.TAX_FORM_1040:
            suggestions.append("Consider extracting tax fields: income, deductions, tax liability")
            suggestions.append("Validate SSN and EIN formats")
        
        elif doc_type in [DocumentType.BANK_STATEMENT, DocumentType.CREDIT_CARD_STATEMENT]:
            suggestions.append("Extract transaction history and account balances")
            suggestions.append("Validate account numbers and amounts")
        
        elif doc_type == DocumentType.INVOICE:
            suggestions.append("Extract invoice details: number, date, amounts, line items")
            suggestions.append("Validate payment terms and due dates")
        
        elif doc_type == DocumentType.LEGAL_CONTRACT:
            suggestions.append("Extract parties, terms, and signature information")
            suggestions.append("Identify key clauses and obligations")
        
        if len(signals) < 3:
            suggestions.append("Limited classification signals - consider additional analysis")
        
        return suggestions
