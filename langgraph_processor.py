
from typing import Dict, List, Any, TypedDict
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from local_ai_engine import AI_ENGINE
from config import CONFIG
import logging

logger = logging.getLogger(__name__)

class DocumentState(TypedDict):
    documents: List[Document]
    patient_assignments: Dict[str, List[int]]
    processing_stage: str
    confidence_scores: Dict[int, float]
    review_queue: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class LangGraphProcessor:
    """LangGraph-based document processing workflow"""
    
    def __init__(self):
        self.graph = self._build_processing_graph()
    
    def _build_processing_graph(self) -> StateGraph:
        """Build the document processing graph"""
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
        """Extract text from documents"""
        logger.info("Processing text extraction node")
        
        # Text extraction already done, update stage
        state["processing_stage"] = "text_extracted"
        state["metadata"]["text_extraction_complete"] = True
        
        return state
    
    def _classify_documents_node(self, state: DocumentState) -> DocumentState:
        """Classify document types"""
        logger.info("Processing document classification node")
        
        for doc in state["documents"]:
            if doc.page_content:
                classification = AI_ENGINE.classify_document_type(doc.page_content)
                doc.metadata["document_type"] = classification["type"]
                doc.metadata["classification_confidence"] = classification["confidence"]
        
        state["processing_stage"] = "documents_classified"
        return state
    
    def _extract_entities_node(self, state: DocumentState) -> DocumentState:
        """Extract named entities from documents"""
        logger.info("Processing entity extraction node")
        
        for doc in state["documents"]:
            if doc.page_content:
                entities = AI_ENGINE.extract_entities(doc.page_content)
                doc.metadata["entities"] = entities
                
                # Extract patient names from entities
                patient_entities = [
                    e for e in entities 
                    if e.get('entity_group') == 'PER' and 
                    e.get('score', 0) >= CONFIG.confidence_threshold
                ]
                
                if patient_entities:
                    # Take highest confidence patient name
                    best_patient = max(patient_entities, key=lambda x: x.get('score', 0))
                    doc.metadata["extracted_patient"] = best_patient['word']
                    doc.metadata["patient_confidence"] = best_patient['score']
        
        state["processing_stage"] = "entities_extracted"
        return state
    
    def _assign_patients_node(self, state: DocumentState) -> DocumentState:
        """Assign pages to patients"""
        logger.info("Processing patient assignment node")
        
        # Use existing assignment logic but enhance with AI validation
        from assigner import assign_pages_to_patients
        from name_extractor import NameExtractor
        
        name_extractor = NameExtractor()
        name_extractor.extract(state["documents"])
        consolidated = name_extractor.get_consolidated_name_tokens(min_pages=1)
        
        assignments, review = assign_pages_to_patients(state["documents"], consolidated)
        
        state["patient_assignments"] = assignments
        state["review_queue"] = [
            {"page": page, "candidates": candidates, "reason": reason}
            for page, candidates, reason in review
        ]
        state["processing_stage"] = "patients_assigned"
        
        return state
    
    def _validate_assignments_node(self, state: DocumentState) -> DocumentState:
        """Validate patient assignments using AI"""
        logger.info("Processing assignment validation node")
        
        validated_assignments = {}
        confidence_scores = {}
        
        for patient, pages in state["patient_assignments"].items():
            validated_pages = []
            
            for page_num in pages:
                # Find the document for this page
                doc = next(
                    (d for d in state["documents"] 
                     if d.metadata.get("page") == page_num), 
                    None
                )
                
                if doc and doc.page_content:
                    confidence = AI_ENGINE.validate_patient_assignment(
                        doc.page_content, patient
                    )
                    confidence_scores[page_num] = confidence
                    
                    # Only keep assignments above threshold
                    if confidence >= CONFIG.confidence_threshold:
                        validated_pages.append(page_num)
                    else:
                        # Move to review queue
                        state["review_queue"].append({
                            "page": page_num,
                            "patient": patient,
                            "confidence": confidence,
                            "reason": "Low AI validation confidence"
                        })
            
            if validated_pages:
                validated_assignments[patient] = validated_pages
        
        state["patient_assignments"] = validated_assignments
        state["confidence_scores"] = confidence_scores
        state["processing_stage"] = "assignments_validated"
        
        return state
    
    def _quality_check_node(self, state: DocumentState) -> DocumentState:
        """Perform quality checks on assignments"""
        logger.info("Processing quality check node")
        
        quality_metrics = {
            "total_pages": len(state["documents"]),
            "assigned_pages": sum(len(pages) for pages in state["patient_assignments"].values()),
            "unassigned_pages": len(state["review_queue"]),
            "average_confidence": sum(state["confidence_scores"].values()) / len(state["confidence_scores"]) if state["confidence_scores"] else 0,
            "patients_identified": len(state["patient_assignments"])
        }
        
        state["metadata"]["quality_metrics"] = quality_metrics
        state["processing_stage"] = "quality_checked"
        
        # Flag potential issues
        if quality_metrics["unassigned_pages"] > quality_metrics["total_pages"] * 0.3:
            logger.warning("High number of unassigned pages detected")
        
        if quality_metrics["average_confidence"] < 0.7:
            logger.warning("Low average confidence in assignments")
        
        return state
    
    def _generate_summaries_node(self, state: DocumentState) -> DocumentState:
        """Generate patient summaries"""
        logger.info("Processing summary generation node")
        
        summaries = {}
        for patient, pages in state["patient_assignments"].items():
            patient_docs = [
                doc for doc in state["documents"]
                if doc.metadata.get("page") in pages
            ]
            
            summary = AI_ENGINE.generate_patient_summary(patient_docs, patient)
            summaries[patient] = summary
        
        state["metadata"]["patient_summaries"] = summaries
        state["processing_stage"] = "summaries_generated"
        
        return state
    
    def process_documents(self, documents: List[Document]) -> DocumentState:
        """Process documents through the complete workflow"""
        initial_state = DocumentState(
            documents=documents,
            patient_assignments={},
            processing_stage="initialized",
            confidence_scores={},
            review_queue=[],
            metadata={}
        )
        
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"LangGraph processing failed: {e}")
            return initial_state

# Global processor instance
LANGGRAPH_PROCESSOR = LangGraphProcessor()
