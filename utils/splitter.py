
"""
Enhanced PDF splitting utility with better error handling
"""
import os
import logging
from pathlib import Path
from typing import Dict, List
import fitz
from langchain.schema import Document


class PDFSplitter:
    """Enhanced PDF splitter with better organization and error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def split(self, documents: List[Document], 
             page_assignments: Dict[int, str], 
             output_dir: str):
        """Split PDF based on page assignments"""
        
        if not documents:
            raise ValueError("No documents provided for splitting")
        
        # Get source PDF path from first document
        source_pdf = documents[0].metadata.get("source_pdf")
        if not source_pdf or not os.path.exists(source_pdf):
            raise ValueError("Source PDF not found")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group pages by patient
        patient_pages: Dict[str, List[int]] = {}
        for page_index, patient in page_assignments.items():
            if patient not in patient_pages:
                patient_pages[patient] = []
            patient_pages[patient].append(page_index)
        
        # Open source PDF
        source_doc = fitz.open(source_pdf)
        
        try:
            for patient, pages in patient_pages.items():
                if not pages:
                    continue
                
                # Clean patient name for filename
                safe_name = self._clean_filename(patient)
                output_file = output_path / f"{safe_name}.pdf"
                
                # Create new PDF for this patient
                new_doc = fitz.open()
                
                # Add pages (convert from 0-based to 1-based)
                for page_index in sorted(pages):
                    if 0 <= page_index < len(source_doc):
                        new_doc.insert_pdf(source_doc, from_page=page_index, to_page=page_index)
                
                # Save patient PDF
                new_doc.save(str(output_file))
                new_doc.close()
                
                self.logger.info(f"Created {output_file} with {len(pages)} pages")
        
        finally:
            source_doc.close()
    
    def _clean_filename(self, name: str) -> str:
        """Clean patient name for use as filename"""
        # Remove invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        cleaned = name
        for char in invalid_chars:
            cleaned = cleaned.replace(char, '_')
        
        # Limit length and clean up
        cleaned = cleaned.strip()[:50]
        return cleaned if cleaned else "Unknown_Patient"
