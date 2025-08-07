
"""
Local AI engine with enhanced model management and progress tracking
"""
import os
import torch
import logging
from typing import List, Dict, Any, Optional, Tuple
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    pipeline, AutoModelForCausalLM
)
from langchain.schema import Document

from core.base import ProgressCallback, ProcessingStage


class LocalAIEngine:
    """Enhanced local AI engine with GPU/CPU support and progress tracking"""
    
    def __init__(self, config, progress_callback: Optional[ProgressCallback] = None):
        self.config = config
        self.progress_callback = progress_callback
        self.device = getattr(config, 'device', 'cpu')
        self.use_gpu = getattr(config, 'use_gpu', False)
        self.local_llm_path = getattr(config, 'local_llm_path', None)
        self.local_ner_model = getattr(config, 'local_ner_model', 'dslim/bert-large-NER')
        self.confidence_threshold = getattr(config, 'confidence_threshold', 0.7)
        
        self.ner_pipeline = None
        self.llm_pipeline = None
        self.classification_pipeline = None
        
        self.logger = logging.getLogger(__name__)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models with progress tracking"""
        try:
            self._update_progress(ProcessingStage.INITIALIZATION, 0.0, "Initializing AI models")
            
            # Initialize NER model
            self._update_progress(ProcessingStage.INITIALIZATION, 0.2, "Loading NER model")
            self._load_ner_model()
            
            # Initialize local LLM if path provided
            if self.local_llm_path:
                self._update_progress(ProcessingStage.INITIALIZATION, 0.5, "Loading local LLM")
                self._load_local_llm()
            
            # Initialize document classifier
            self._update_progress(ProcessingStage.INITIALIZATION, 0.8, "Loading document classifier")
            self._load_classifier()
            
            self._update_progress(ProcessingStage.INITIALIZATION, 1.0, 
                                f"Models initialized on device: {self.device}")
            self.logger.info(f"Models initialized on device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            self._fallback_to_cpu()
    
    def _update_progress(self, stage: ProcessingStage, progress: float, message: str):
        """Update progress if callback is available"""
        if self.progress_callback:
            self.progress_callback.update(stage, progress, message)
    
    def _load_ner_model(self):
        """Load NER model with GPU/CPU support"""
        try:
            device_id = 0 if self.use_gpu else -1
            
            self.ner_pipeline = pipeline(
                "ner",
                model=self.local_ner_model,
                aggregation_strategy="simple",
                device=device_id,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32
            )
            self.logger.info("NER model loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load NER model: {e}")
            self.ner_pipeline = None
    
    def _load_local_llm(self):
        """Load local LLM (20B parameter model)"""
        try:
            if not self.local_llm_path or not os.path.exists(self.local_llm_path):
                self.logger.warning("Local LLM path not found, skipping LLM initialization")
                return
            
            torch_dtype = torch.float16 if self.use_gpu else torch.float32
            
            tokenizer = AutoTokenizer.from_pretrained(self.local_llm_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.local_llm_path,
                torch_dtype=torch_dtype,
                device_map="auto" if self.use_gpu else None,
                low_cpu_mem_usage=True
            )
            
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.use_gpu else -1
            )
            self.logger.info("Local LLM loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load local LLM: {e}")
            self.llm_pipeline = None
    
    def _load_classifier(self):
        """Load document type classifier"""
        try:
            self.classification_pipeline = pipeline(
                "text-classification",
                model="emilyalsentzer/Bio_ClinicalBERT",
                device=0 if self.use_gpu else -1
            )
            self.logger.info("Document classifier loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load classifier: {e}")
            self.classification_pipeline = None
    
    def _fallback_to_cpu(self):
        """Fallback to CPU-only processing"""
        self.logger.warning("Falling back to CPU-only processing")
        self.use_gpu = False
        self.device = "cpu"
        self._initialize_models()
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status information"""
        return {
            "device": self.device,
            "gpu_available": self.use_gpu,
            "ner_loaded": self.ner_pipeline is not None,
            "llm_loaded": self.llm_pipeline is not None,
            "classifier_loaded": self.classification_pipeline is not None,
            "ner_model": self.local_ner_model,
            "llm_path": self.local_llm_path
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities with confidence scores"""
        if not self.ner_pipeline:
            # Fallback to simple name extraction
            import re
            names = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text)
            return [
                {'word': name, 'entity_group': 'PER', 'score': 0.8} 
                for name in names[:5]
            ]
        
        try:
            entities = self.ner_pipeline(text)
            filtered = [
                e for e in entities 
                if e.get('score', 0) >= self.confidence_threshold
            ]
            return filtered
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return []
    
    def classify_document_type(self, text: str) -> Dict[str, Any]:
        """Classify document type"""
        if not self.classification_pipeline:
            return {"type": "unknown", "confidence": 0.0}
        
        try:
            result = self.classification_pipeline(text[:512])
            return {
                "type": result[0]['label'],
                "confidence": result[0]['score']
            }
        except Exception as e:
            self.logger.error(f"Document classification failed: {e}")
            return {"type": "unknown", "confidence": 0.0}
    
    def generate_patient_summary(self, documents: List[Document], patient_name: str) -> str:
        """Generate patient summary using local LLM"""
        if not self.llm_pipeline:
            return f"Summary generation unavailable for {patient_name}"
        
        try:
            content = "\n".join([
                doc.page_content[:200] for doc in documents 
                if doc.metadata.get('patient_name') == patient_name
            ][:5])
            
            prompt = f"""
            Based on the following medical records for patient {patient_name}, 
            provide a brief summary of key medical information:
            
            {content}
            
            Summary:
            """
            
            result = self.llm_pipeline(
                prompt, 
                max_length=200, 
                num_return_sequences=1,
                temperature=0.3
            )
            
            return result[0]['generated_text'].split("Summary:")[-1].strip()
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return f"Summary generation failed for {patient_name}"
    
    def validate_patient_assignment(self, page_content: str, patient_name: str) -> float:
        """Use LLM to validate patient-page assignments"""
        if not self.llm_pipeline:
            return 0.5
        
        try:
            prompt = f"""
            Does this medical document belong to patient "{patient_name}"?
            
            Document content: {page_content[:300]}
            
            Answer with confidence score (0.0-1.0):
            """
            
            result = self.llm_pipeline(prompt, max_length=50, temperature=0.1)
            
            response = result[0]['generated_text']
            import re
            match = re.search(r'([0-1]\.?\d*)', response)
            if match:
                return float(match.group(1))
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return 0.5
from typing import Dict, Any, List
import logging


class LocalAIEngine:
    """Local AI engine for document processing"""
    
    def __init__(self, config, progress_tracker=None):
        self.config = config
        self.progress_tracker = progress_tracker
        self.logger = logging.getLogger(__name__)
    
    def classify_document_type(self, content: str) -> Dict[str, Any]:
        """Classify document type"""
        # Simple mock classification
        if "medical" in content.lower() or "patient" in content.lower():
            return {"type": "medical_record", "confidence": 0.9}
        else:
            return {"type": "unknown", "confidence": 0.5}
    
    def extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from content"""
        # Simple mock entity extraction
        import re
        
        entities = []
        
        # Look for potential names
        name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        names = re.findall(name_pattern, content)
        
        for name in names:
            entities.append({
                "text": name,
                "type": "PERSON",
                "confidence": 0.8
            })
        
        return entities
