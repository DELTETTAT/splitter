
"""
Enhanced name extraction with NER and fuzzy matching
"""
import re
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Set, Optional
from langchain.schema import Document
from rapidfuzz import fuzz

from core.base import BaseExtractor, ProcessingResult, ProcessingStage, ProgressCallback

try:
    from transformers import pipeline
    NER_AVAILABLE = True
except ImportError:
    NER_AVAILABLE = False


class NameExtractor(BaseExtractor):
    """Enhanced name extractor with NER and pattern matching"""
    
    NAME_PATTERNS = [
        re.compile(r"(?:Customer Name|Patient Name)[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE),
        re.compile(r"Customer[:\-]?\s*([A-Z][A-Z\s]+),\s*([A-Z][A-Z\s]+)", re.IGNORECASE),
        re.compile(r"DELIVER TO\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE),
        re.compile(r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'),
        re.compile(r'(?:Patient|Name|DOB|Date of Birth)[\s:]+([A-Z][a-z]+ [A-Z][a-z]+)'),
        re.compile(r'([A-Z]{2,}(?:\s+[A-Z]{2,})+)'),
    ]
    
    NON_PATIENT_KEYWORDS = [
        "Medical", "Equipment", "System", "Service", "Group",
        "Confirmation", "Order", "Setup", "Phone", "Progress",
        "Encounter", "Digitally", "Signed", "Date", "Lite",
        "Respiratory", "Check", "Care", "Home", "Hospice",
        "Medication", "Oxygen", "Notes"
    ]
    
    def __init__(self, config, progress_callback: Optional[ProgressCallback] = None):
        super().__init__(config)
        self.progress_callback = progress_callback
        self.name_counter = Counter()
        self.full_name_registry = {}
        self.token_map = {}
        self.extracted_names = defaultdict(set)
        self._ner_pipeline = None
        self._initialize_ner()
    
    def _initialize_ner(self):
        """Initialize NER pipeline if available"""
        if NER_AVAILABLE:
            try:
                self._ner_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-large-NER",
                    aggregation_strategy="simple",
                    device=-1
                )
                self.logger.info("NER pipeline initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NER: {e}")
        else:
            self.logger.warning("Transformers not available, using regex only")
    
    def extract(self, documents: List[Document]) -> ProcessingResult:
        """Extract names from all documents with progress tracking"""
        start_time = time.time()
        self.extracted_names.clear()
        self.name_counter.clear()
        self.full_name_registry.clear()
        self.token_map.clear()
        
        total_docs = len(documents)
        self._update_progress(ProcessingStage.NAME_EXTRACTION, 0.0, 
                            f"Starting name extraction from {total_docs} documents")
        
        for i, doc in enumerate(documents):
            page_num = doc.metadata.get('page', i + 1)
            
            # Extract from structured data first
            patient_name = self._extract_from_structured(doc)
            
            # If no structured name, extract from text
            if not patient_name:
                patient_name = self._extract_from_text(doc.page_content)
            
            # Register the name
            if patient_name:
                self._register_name(patient_name, page_num)
                doc.metadata["patient_name"] = patient_name
            
            # Update progress
            progress = (i + 1) / total_docs
            self._update_progress(ProcessingStage.NAME_EXTRACTION, progress,
                                f"Processed page {i + 1}/{total_docs}")
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            success=True,
            stage=ProcessingStage.NAME_EXTRACTION,
            data={
                "extracted_names": dict(self.extracted_names),
                "name_counter": dict(self.name_counter),
                "full_name_registry": self.full_name_registry,
                "token_map": self.token_map
            },
            metadata={"processing_time": processing_time, "total_documents": total_docs},
            errors=[],
            warnings=[],
            processing_time=processing_time
        )
    
    def _update_progress(self, stage: ProcessingStage, progress: float, message: str):
        """Update progress if callback is available"""
        if self.progress_callback:
            self.progress_callback.update(stage, progress, message)
    
    def _extract_from_structured(self, doc: Document) -> Optional[str]:
        """Extract patient name from structured metadata"""
        return doc.metadata.get("structured_data", {}).get("patient")
    
    def _extract_from_text(self, text: str) -> Optional[str]:
        """Extract patient name from text using patterns and NER"""
        if not text:
            return None
        
        # Try regex patterns first
        for pattern in self.NAME_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    clean_name = " ".join(part.strip().title() for part in match if part.strip())
                else:
                    clean_name = self._normalize_name(match)
                
                if self._is_valid_name(clean_name):
                    self.logger.debug(f"Found name via regex: {clean_name}")
                    return clean_name
        
        # Try NER if available
        if self._ner_pipeline:
            try:
                entities = self._ner_pipeline(text[:512])  # Limit text length
                persons = [e for e in entities if e.get('entity_group') == 'PER']
                
                if persons:
                    best = max(persons, key=lambda e: e.get('score', 0))
                    clean_name = self._normalize_name(best['word'])
                    
                    if self._is_valid_name(clean_name):
                        self.logger.debug(f"Found name via NER: {clean_name}")
                        return clean_name
            
            except Exception as e:
                self.logger.warning(f"NER extraction failed: {e}")
        
        return None
    
    def _normalize_name(self, name: str) -> Optional[str]:
        """Normalize and clean name"""
        if not name:
            return None
        
        # Clean whitespace and normalize case
        cleaned = " ".join(name.split()).title()
        
        # Remove common non-name words
        exclude_words = {'The', 'And', 'Or', 'Of', 'In', 'On', 'At', 'To', 'For'}
        words = [w for w in cleaned.split() if w not in exclude_words]
        
        return " ".join(words) if words else None
    
    def _is_valid_name(self, name: str) -> bool:
        """Validate if extracted text is likely a patient name"""
        if not name or len(name) < 3 or len(name) > 50:
            return False
        
        # Check for non-patient keywords
        for keyword in self.NON_PATIENT_KEYWORDS:
            if keyword.lower() in name.lower():
                return False
        
        # Must have at least 2 parts
        parts = name.split()
        if len(parts) < 2:
            return False
        
        # Each part should be reasonable length
        for part in parts:
            if len(part) < 2 or len(part) > 20:
                return False
        
        # Should not contain numbers or special characters
        if re.search(r'[0-9@#$%^&*()_+=\[\]{}|;:,.<>?/~`]', name):
            return False
        
        return True
    
    def _register_name(self, name: str, page_num: int):
        """Register name in internal data structures"""
        self.extracted_names[page_num].add(name)
        
        initials = "".join(w[0] for w in name.split())
        current = self.full_name_registry.get(initials)
        
        if not current or len(name) > len(current):
            self.full_name_registry[initials] = name
        
        self.name_counter[name] += 1
        self.token_map[name] = name.split()
    
    def get_consolidated_name_tokens(self, min_pages: int = 1) -> Dict[str, Set[str]]:
        """Get consolidated names with fuzzy matching"""
        # Group similar names
        name_groups = defaultdict(set)
        all_names = set()
        
        for page_names in self.extracted_names.values():
            all_names.update(page_names)
        
        processed = set()
        
        for name in all_names:
            if name in processed:
                continue
            
            similar_names = {name}
            for other_name in all_names:
                if other_name != name and other_name not in processed:
                    similarity = fuzz.ratio(name.lower(), other_name.lower())
                    if similarity > 85:
                        similar_names.add(other_name)
            
            canonical = max(similar_names, key=len)
            name_groups[canonical] = similar_names
            processed.update(similar_names)
        
        # Filter by minimum page count and convert to token format
        filtered_groups = {}
        for canonical, variants in name_groups.items():
            page_count = sum(
                1 for page_names in self.extracted_names.values()
                if any(variant in page_names for variant in variants)
            )
            if page_count >= min_pages:
                # Convert to token list format for compatibility
                filtered_groups[canonical] = canonical.split()
        
        return filtered_groups


# Import time for timing
import time
from typing import List, Dict, Any
import logging
from langchain.schema import Document

from core.base import BaseExtractor, ProcessingResult, ProcessingStage


class NameExtractor(BaseExtractor):
    """Extract patient names from documents"""
    
    def __init__(self, config, progress_tracker=None):
        super().__init__(config)
        self.progress_tracker = progress_tracker
        self.name_tokens = {}
    
    def extract(self, documents: List[Document]) -> ProcessingResult:
        """Extract names from documents"""
        self.logger.info(f"Extracting names from {len(documents)} documents")
        
        name_data = {}
        for i, doc in enumerate(documents):
            # Simple name extraction logic - can be enhanced
            content = doc.page_content.lower()
            page_num = doc.metadata.get("page", i + 1)
            
            # Look for common patterns
            names = self._extract_names_from_text(content)
            if names:
                name_data[page_num] = names
        
        return ProcessingResult(
            success=True,
            stage=ProcessingStage.NAME_EXTRACTION,
            data={"names": name_data},
            metadata={},
            errors=[],
            warnings=[],
            processing_time=0.0
        )
    
    def process(self, data: Any) -> ProcessingResult:
        """Process method required by BaseProcessor"""
        if isinstance(data, list):
            return self.extract(data)
        else:
            return ProcessingResult(
                success=False,
                stage=ProcessingStage.NAME_EXTRACTION,
                data={},
                metadata={},
                errors=["Invalid input data type"],
                warnings=[],
                processing_time=0.0
            )
    
    def _extract_names_from_text(self, text: str) -> List[str]:
        """Extract potential patient names from text"""
        import re
        
        # Simple pattern matching for names
        name_patterns = [
            r'patient[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'name[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+,\s+[A-Z][a-z]+)',
        ]
        
        names = []
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            names.extend(matches)
        
        return list(set(names))  # Remove duplicates
    
    def get_consolidated_name_tokens(self, min_pages: int = 1) -> Dict[str, List[str]]:
        """Get consolidated name tokens"""
        # Simple consolidation logic
        consolidated = {}
        for name in self.name_tokens:
            if len(self.name_tokens[name]) >= min_pages:
                consolidated[name] = self.name_tokens[name]
        return consolidated
