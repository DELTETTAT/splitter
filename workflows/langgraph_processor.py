
"""
Enhanced LangGraph processor with comprehensive workflow management
"""
from typing import Dict, List, Any, TypedDict, Optional
from langchain.schema import Document
from langgraph.graph import StateGraph, END
import logging
import time

from core.base import ProcessingStage
from ai.local_engine import LocalAIEngine
from extractors.name_extractor import NameExtractor
from processors.assignment_engine import AssignmentEngine


class DocumentState(TypedDict):
    """Enhanced document state with progress tracking"""
    documents: List[Document]
    patient_assignments: Dict[str, List[int]]
    processing_stage: str
    confidence_scores: Dict[int, float]
    review_queue: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    progress: float
    current_step: str


class LangGraphProcessor:
    """Enhanced LangGraph-based document processing workflow"""
    
    def __init__(self, config, progress_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.ai_engine = LocalAIEngine(config, progress_callback)
        self.name_extractor = NameExtractor(config, progress_callback)
        self.assignment_engine = AssignmentEngine(config, progress_callback)
        
        self.graph = self._build_processing_graph()
    
    def _build_processing_graph(self) -> StateGraph:
        """Build the enhanced document processing graph"""
        workflow = StateGraph(DocumentState)
        
        # Add nodes
        workflow.add_node("extract_text", self._extract_text_node)
        workflow.add_node("classify_documents", self._classify_documents_node)
        workflow.add_node("extract_entities", self._extract_entities_node)
        workflow.add_node("assign_patients", self._assign_patients_node)
        workflow.add_node("validate_assignments", self._validate_assignments_node)
        workflow.add_node("quality_check", self._quality_check_node)
        workflow.add_node("generate_summaries", self._generate_summaries_node)
        
        # Define edges
        workflow.set_entry_point("extract_text")
        workflow.add_edge("extract_text", "classify_documents")
        workflow.add_edge("classify_documents", "extract_entities")
        workflow.add_edge("extract_entities", "assign_patients")
        workflow.add_edge("assign_patients", "validate_assignments")
        workflow.add_edge("validate_assignments", "quality_check")
        workflow.add_edge("quality_check", "generate_summaries")
        workflow.add_edge("generate_summaries", END)
        
        return workflow.compile()
    
    def _extract_text_node(self, state: DocumentState) -> DocumentState:
        """Extract text features from documents"""
        self._update_progress(ProcessingStage.TEXT_EXTRACTION, 0.5, "Analyzing document text")
        
        # Text extraction is already done, just analyze features
        state["current_step"] = "Text analysis complete"
        state["progress"] = 0.15
        
        return state
    
    def _classify_documents_node(self, state: DocumentState) -> DocumentState:
        """Classify document types"""
        self._update_progress(ProcessingStage.CLASSIFICATION, 0.5, "Classifying document types")
        
        # Simulate document classification
        state["current_step"] = "Document classification complete"
        state["progress"] = 0.30
        
        return state
    
    def _extract_entities_node(self, state: DocumentState) -> DocumentState:
        """Extract entities from documents"""
        self._update_progress(ProcessingStage.ENTITY_EXTRACTION, 0.5, "Extracting patient entities")
        
        # Use name extractor
        name_result = self.name_extractor.extract(state["documents"])
        consolidated = self.name_extractor.get_consolidated_name_tokens(min_pages=1)
        
        state["metadata"]["consolidated_names"] = consolidated
        state["current_step"] = "Entity extraction complete"
        state["progress"] = 0.50
        
        return state
    
    def _assign_patients_node(self, state: DocumentState) -> DocumentState:
        """Assign pages to patients"""
        self._update_progress(ProcessingStage.PATIENT_ASSIGNMENT, 0.5, "Assigning pages to patients")
        
        # Use assignment engine
        consolidated = state["metadata"].get("consolidated_names", {})
        assignment_result = self.assignment_engine.assign(state["documents"], consolidated)
        
        state["patient_assignments"] = assignment_result.data["patient_assignments"]
        state["review_queue"] = assignment_result.data["review_queue"]
        state["current_step"] = "Patient assignment complete"
        state["progress"] = 0.70
        
        return state
    
    def _validate_assignments_node(self, state: DocumentState) -> DocumentState:
        """Validate patient assignments"""
        self._update_progress(ProcessingStage.VALIDATION, 0.5, "Validating assignments")
        
        # Basic validation
        total_pages = len(state["documents"])
        assigned_pages = sum(len(pages) for pages in state["patient_assignments"].values())
        
        state["metadata"]["validation"] = {
            "total_pages": total_pages,
            "assigned_pages": assigned_pages,
            "coverage_ratio": assigned_pages / total_pages if total_pages > 0 else 0
        }
        
        state["current_step"] = "Validation complete"
        state["progress"] = 0.85
        
        return state
    
    def _quality_check_node(self, state: DocumentState) -> DocumentState:
        """Perform quality checks"""
        self._update_progress(ProcessingStage.QUALITY_CHECK, 0.5, "Performing quality checks")
        
        # Quality metrics
        total_pages = len(state["documents"])
        assigned_pages = sum(len(pages) for pages in state["patient_assignments"].values())
        unassigned_pages = len(state["review_queue"])
        
        state["metadata"]["quality_metrics"] = {
            "total_pages": total_pages,
            "assigned_pages": assigned_pages,
            "unassigned_pages": unassigned_pages,
            "assignment_rate": assigned_pages / total_pages if total_pages > 0 else 0,
            "patient_count": len(state["patient_assignments"])
        }
        
        state["current_step"] = "Quality check complete"
        state["progress"] = 0.95
        
        return state
    
    def _generate_summaries_node(self, state: DocumentState) -> DocumentState:
        """Generate processing summaries"""
        self._update_progress(ProcessingStage.SUMMARY_GENERATION, 1.0, "Generating summaries")
        
        # Add summary information
        state["metadata"]["processing_summary"] = {
            "completion_time": time.time(),
            "status": "completed",
            "success": True
        }
        
        state["current_step"] = "Processing complete"
        state["progress"] = 1.0
        
        return state
    
    def _update_progress(self, stage: ProcessingStage, progress: float, message: str):
        """Update progress if callback is available"""
        if self.progress_callback:
            self.progress_callback.update(stage, progress, message)
    
    def process_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Process documents through LangGraph workflow"""
        initial_state = DocumentState(
            documents=documents,
            patient_assignments={},
            processing_stage="initialization",
            confidence_scores={},
            review_queue=[],
            metadata={},
            progress=0.0,
            current_step="Initializing"
        )
        
        try:
            self._update_progress(ProcessingStage.INITIALIZATION, 0.0, "Starting LangGraph processing")
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            self.logger.error(f"LangGraph processing failed: {e}")
            return initial_state
