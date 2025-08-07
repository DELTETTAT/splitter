
"""
Base classes and interfaces for the medical record processing system
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from dataclasses import dataclass
from enum import Enum


class ProcessingStage(Enum):
    """Enumeration of processing stages"""
    INITIALIZATION = "initialization"
    TEXT_EXTRACTION = "text_extraction"
    NAME_EXTRACTION = "name_extraction"
    CLASSIFICATION = "classification"
    ENTITY_EXTRACTION = "entity_extraction"
    PATIENT_ASSIGNMENT = "patient_assignment"
    VALIDATION = "validation"
    QUALITY_CHECK = "quality_check"
    SUMMARY_GENERATION = "summary_generation"
    COMPLETE = "complete"


@dataclass
class ProcessingResult:
    """Standardized processing result structure"""
    success: bool
    stage: ProcessingStage
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    processing_time: float


@dataclass
class PatientAssignment:
    """Standardized patient assignment structure"""
    patient_name: str
    pages: List[int]
    confidence_scores: Dict[int, float]
    metadata: Dict[str, Any]


class BaseProcessor(ABC):
    """Base class for all processors"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = self._setup_logger()
    
    @abstractmethod
    def process(self, data: Any) -> ProcessingResult:
        """Process the input data and return results"""
        pass
    
    def _setup_logger(self):
        """Setup logger for the processor"""
        import logging
        return logging.getLogger(self.__class__.__name__)


class BaseExtractor(BaseProcessor):
    """Base class for all extractors"""
    
    @abstractmethod
    def extract(self, documents: List[Document]) -> ProcessingResult:
        """Extract data from documents"""
        pass


class BaseAssigner(BaseProcessor):
    """Base class for assignment engines"""
    
    @abstractmethod
    def assign(self, documents: List[Document], 
              extracted_data: Dict[str, Any]) -> ProcessingResult:
        """Assign pages to patients"""
        pass


class ProgressCallback:
    """Callback interface for progress tracking"""
    
    def __init__(self):
        self.callbacks = []
    
    def add_callback(self, callback):
        """Add a progress callback function"""
        self.callbacks.append(callback)
    
    def update(self, stage: ProcessingStage, progress: float, 
               message: str = "", metadata: Dict[str, Any] = None):
        """Update progress for all registered callbacks"""
        for callback in self.callbacks:
            callback(stage, progress, message, metadata or {})
