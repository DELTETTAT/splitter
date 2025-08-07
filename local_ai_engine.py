
import os
import torch
import logging
from typing import List, Dict, Any, Optional, Tuple
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    pipeline, AutoModelForCausalLM
)
from langchain.schema import Document
from config import CONFIG

logger = logging.getLogger(__name__)

class LocalAIEngine:
    """Local AI engine with GPU/CPU support for medical record processing"""
    
    def __init__(self):
        self.device = CONFIG.device
        self.ner_pipeline = None
        self.llm_pipeline = None
        self.classification_pipeline = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models based on available hardware"""
        try:
            # Initialize NER model
            self._load_ner_model()
            
            # Initialize local LLM if path provided
            if CONFIG.local_llm_path:
                self._load_local_llm()
            
            # Initialize document classifier
            self._load_classifier()
            
            logger.info(f"Models initialized on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self._fallback_to_cpu()
    
    def _load_ner_model(self):
        """Load NER model with GPU/CPU support"""
        try:
            model_config = CONFIG.get_model_config()
            device_id = 0 if CONFIG.use_gpu else -1
            
            self.ner_pipeline = pipeline(
                "ner",
                model=CONFIG.local_ner_model,
                aggregation_strategy="simple",
                device=device_id,
                torch_dtype=torch.float16 if CONFIG.use_gpu else torch.float32
            )
            logger.info("NER model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load NER model: {e}")
            self.ner_pipeline = None
    
    def _load_local_llm(self):
        """Load local LLM (20B parameter model)"""
        try:
            if not CONFIG.local_llm_path or not os.path.exists(CONFIG.local_llm_path):
                logger.warning("Local LLM path not found, skipping LLM initialization")
                return
            
            # Load with appropriate precision based on hardware
            torch_dtype = torch.float16 if CONFIG.use_gpu else torch.float32
            
            tokenizer = AutoTokenizer.from_pretrained(CONFIG.local_llm_path)
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG.local_llm_path,
                torch_dtype=torch_dtype,
                device_map="auto" if CONFIG.use_gpu else None,
                low_cpu_mem_usage=True
            )
            
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if CONFIG.use_gpu else -1
            )
            logger.info("Local LLM loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            self.llm_pipeline = None
    
    def _load_classifier(self):
        """Load document type classifier"""
        try:
            # Use a medical document classifier or general classifier
            self.classification_pipeline = pipeline(
                "text-classification",
                model="emilyalsentzer/Bio_ClinicalBERT",
                device=0 if CONFIG.use_gpu else -1
            )
            logger.info("Document classifier loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load classifier: {e}")
            self.classification_pipeline = None
    
    def _fallback_to_cpu(self):
        """Fallback to CPU-only processing"""
        logger.warning("Falling back to CPU-only processing")
        CONFIG.use_gpu = False
        CONFIG.device = "cpu"
        self._initialize_models()
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities with confidence scores"""
        if not self.ner_pipeline:
            # Fallback to simple name extraction
            import re
            names = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text)
            return [
                {'word': name, 'entity_group': 'PER', 'score': 0.8} 
                for name in names[:5]  # Limit to 5 names
            ]
        
        try:
            entities = self.ner_pipeline(text)
            # Filter by confidence threshold
            filtered = [
                e for e in entities 
                if e.get('score', 0) >= CONFIG.confidence_threshold
            ]
            return filtered
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def classify_document_type(self, text: str) -> Dict[str, Any]:
        """Classify document type (lab result, prescription, etc.)"""
        if not self.classification_pipeline:
            return {"type": "unknown", "confidence": 0.0}
        
        try:
            result = self.classification_pipeline(text[:512])  # Limit text length
            return {
                "type": result[0]['label'],
                "confidence": result[0]['score']
            }
        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            return {"type": "unknown", "confidence": 0.0}
    
    def generate_patient_summary(self, documents: List[Document], patient_name: str) -> str:
        """Generate patient summary using local LLM"""
        if not self.llm_pipeline:
            return f"Summary generation unavailable for {patient_name}"
        
        try:
            # Combine relevant document content
            content = "\n".join([
                doc.page_content[:200] for doc in documents 
                if doc.metadata.get('patient_name') == patient_name
            ][:5])  # Limit to 5 documents
            
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
            logger.error(f"Summary generation failed: {e}")
            return f"Summary generation failed for {patient_name}"
    
    def validate_patient_assignment(self, page_content: str, patient_name: str) -> float:
        """Use LLM to validate patient-page assignments"""
        if not self.llm_pipeline:
            return 0.5  # Default confidence
        
        try:
            prompt = f"""
            Does this medical document belong to patient "{patient_name}"?
            
            Document content: {page_content[:300]}
            
            Answer with confidence score (0.0-1.0):
            """
            
            result = self.llm_pipeline(prompt, max_length=50, temperature=0.1)
            
            # Extract confidence score from response
            response = result[0]['generated_text']
            # Simple regex to extract decimal between 0-1
            import re
            match = re.search(r'([0-1]\.?\d*)', response)
            if match:
                return float(match.group(1))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return 0.5

# Global AI engine instance
AI_ENGINE = LocalAIEngine()
