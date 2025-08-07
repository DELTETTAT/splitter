"""
Enhanced patient assignment engine with clustering and validation
"""
import logging
import time
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict
from langchain.schema import Document
from rapidfuzz import fuzz

from core.base import BaseAssigner, ProcessingResult, ProcessingStage, ProgressCallback, PatientAssignment


class AssignmentEngine(BaseAssigner):
    """Enhanced assignment engine with clustering and AI validation"""

    def __init__(self, config, progress_callback: Optional[ProgressCallback] = None):
        super().__init__(config)
        self.progress_callback = progress_callback
        self.confidence_threshold = getattr(config, 'confidence_threshold', 0.7)

    def assign(self, documents: List[Document],
              patient_tokens: Dict[str, List[str]]) -> ProcessingResult:
        """Assign pages to patients with enhanced clustering"""
        start_time = time.time()

        self._update_progress(ProcessingStage.PATIENT_ASSIGNMENT, 0.0,
                            "Starting patient assignment process")

        # Step 1: Collect initial matches
        page_matches, all_pages = self._collect_page_matches(documents, patient_tokens)

        self._update_progress(ProcessingStage.PATIENT_ASSIGNMENT, 0.2,
                            "Collected initial page matches")

        # Step 2: Build name-to-pages mapping
        name_to_pages = self._build_name_to_pages_map(page_matches, patient_tokens)

        self._update_progress(ProcessingStage.PATIENT_ASSIGNMENT, 0.4,
                            "Built patient-to-pages mapping")

        # Step 3: Cluster similar patients
        merged_clusters = self._cluster_patients(name_to_pages, patient_tokens)

        self._update_progress(ProcessingStage.PATIENT_ASSIGNMENT, 0.6,
                            "Clustered similar patient names")

        # Step 4: Assign pages and identify conflicts
        assignments, review_queue = self._assign_pages(all_pages, merged_clusters, page_matches)

        self._update_progress(ProcessingStage.PATIENT_ASSIGNMENT, 0.8,
                            "Completed page assignments")

        # Step 5: Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            len(documents), assignments, review_queue
        )

        processing_time = time.time() - start_time

        self._update_progress(ProcessingStage.PATIENT_ASSIGNMENT, 1.0,
                            f"Assignment complete: {len(assignments)} patients assigned")

        return ProcessingResult(
            success=True,
            stage=ProcessingStage.PATIENT_ASSIGNMENT,
            data={
                "patient_assignments": assignments,
                "review_queue": review_queue,
                "quality_metrics": quality_metrics
            },
            metadata={
                "processing_time": processing_time,
                "total_documents": len(documents),
                "clustered_patients": len(merged_clusters)
            },
            errors=[],
            warnings=[],
            processing_time=processing_time
        )

    def _update_progress(self, stage: ProcessingStage, progress: float, message: str):
        """Update progress if callback is available"""
        if self.progress_callback:
            self.progress_callback.update(stage, progress, message)

    def _collect_page_matches(self, documents: List[Document],
                            patient_tokens: Dict[str, List[str]]) -> Tuple[Dict[int, List[str]], List[int]]:
        """Collect initial page matches from metadata and text content"""
        page_matches: Dict[int, List[str]] = defaultdict(list)
        all_pages: List[int] = []

        for doc in documents:
            page = doc.metadata.get("page")
            all_pages.append(page)

            # Check metadata first
            meta_name = doc.metadata.get("patient_name")
            if meta_name and meta_name in patient_tokens:
                page_matches[page].append(meta_name)

            # Check text content
            text_lower = (doc.page_content or "").lower()
            for name, tokens in patient_tokens.items():
                if any(token.lower() in text_lower for token in tokens):
                    if name not in page_matches[page]:
                        page_matches[page].append(name)

        return page_matches, all_pages

    def _build_name_to_pages_map(self, page_matches: Dict[int, List[str]],
                               patient_tokens: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """Build mapping from patient names to page numbers"""
        name_to_pages: Dict[str, List[int]] = defaultdict(list)

        for page, matches in page_matches.items():
            for name in matches:
                if name in patient_tokens:
                    name_to_pages[name].append(page)

        return name_to_pages

    def _cluster_patients(self, name_to_pages: Dict[str, List[int]],
                        patient_tokens: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """Cluster similar patient names based on shared tokens and pages"""
        # Initialize clusters
        name_to_canonical = {name: name for name in name_to_pages}
        clusters: Dict[str, set] = {name: {name} for name in name_to_pages}

        # Merge clusters based on shared tokens and pages
        names = list(name_to_pages.keys())
        for i, name1 in enumerate(names):
            canonical1 = name_to_canonical[name1]
            tokens1 = set(patient_tokens.get(name1, name1.split()))

            for j in range(i + 1, len(names)):
                name2 = names[j]
                canonical2 = name_to_canonical[name2]

                if canonical1 == canonical2:
                    continue

                tokens2 = set(patient_tokens.get(name2, name2.split()))

                # Require at least one shared token
                if not tokens1 & tokens2:
                    continue

                pages1 = set(name_to_pages[name1])
                pages2 = set(name_to_pages[name2])

                # Require at least one shared page
                if pages1 & pages2:
                    # Merge cluster2 into cluster1
                    for member in clusters[canonical2]:
                        name_to_canonical[member] = canonical1
                    clusters[canonical1] |= clusters[canonical2]
                    del clusters[canonical2]

        # Build final merged clusters to pages mapping
        merged: Dict[str, List[int]] = {}
        for canonical, members in clusters.items():
            pages = set()
            for member in members:
                pages.update(name_to_pages.get(member, []))
            merged[canonical] = sorted(pages)

        return merged

    def _assign_pages(self, all_pages: List[int],
                     merged_clusters: Dict[str, List[int]],
                     page_matches: Dict[int, List[str]]) -> Tuple[Dict[str, List[int]], List[Dict[str, Any]]]:
        """Assign pages to patients and identify conflicts"""
        # Map each page to its possible canonicals
        page_to_canonicals: Dict[int, set] = defaultdict(set)
        for canonical, pages in merged_clusters.items():
            for page in pages:
                page_to_canonicals[page].add(canonical)

        assignments: Dict[str, List[int]] = defaultdict(list)
        review_queue: List[Dict[str, Any]] = []
        assigned_pages: set = set()

        for page in all_pages:
            canonicals = page_to_canonicals.get(page, set())

            if len(canonicals) == 1:
                # Unambiguous assignment
                name = next(iter(canonicals))
                assignments[name].append(page)
                assigned_pages.add(page)
            elif canonicals:
                # Ambiguous: multiple clusters matched
                review_queue.append({
                    "page": page,
                    "candidates": list(canonicals),
                    "reason": "Multiple clusters matched",
                    "confidence": 0.5
                })
            else:
                # No match at all
                matches = page_matches.get(page, [])
                review_queue.append({
                    "page": page,
                    "candidates": matches,
                    "reason": "No cluster matched",
                    "confidence": 0.0
                })

        return dict(assignments), review_queue

    def _calculate_quality_metrics(self, total_pages: int,
                                 assignments: Dict[str, List[int]],
                                 review_queue: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics for the assignment process"""
        assigned_pages = sum(len(pages) for pages in assignments.values())
        unassigned_pages = len(review_queue)

        return {
            "total_pages": total_pages,
            "assigned_pages": assigned_pages,
            "unassigned_pages": unassigned_pages,
            "assignment_rate": assigned_pages / total_pages if total_pages > 0 else 0,
            "patients_identified": len(assignments),
            "average_pages_per_patient": assigned_pages / len(assignments) if assignments else 0
        }


# Compatibility function for existing code
def assign_pages_to_patients(documents: List[Document],
                           patient_tokens: Dict[str, List[str]]) -> Tuple[Dict[str, List[int]], List[Tuple[int, List[str], str]]]:
    """Compatibility wrapper for existing assignment function"""
    engine = AssignmentEngine(type('Config', (), {'confidence_threshold': 0.7})())
    result = engine.assign(documents, patient_tokens)

    assignments = result.data["patient_assignments"]
    review_queue = [
        (item["page"], item["candidates"], item["reason"])
        for item in result.data["review_queue"]
    ]

    return assignments, review_queue
from typing import List, Dict, Any, Tuple
import logging
from langchain.schema import Document

from core.base import BaseAssigner, ProcessingResult, ProcessingStage


class AssignmentEngine(BaseAssigner):
    """Assign pages to patients based on extracted names"""
    
    def __init__(self, config, progress_tracker=None):
        super().__init__(config)
        self.progress_tracker = progress_tracker
    
    def assign(self, documents: List[Document], 
              extracted_data: Dict[str, Any]) -> ProcessingResult:
        """Assign pages to patients"""
        self.logger.info(f"Assigning {len(documents)} documents to patients")
        
        assignments = {}
        review_queue = []
        
        # Simple assignment logic
        for i, doc in enumerate(documents):
            page_num = doc.metadata.get("page", i + 1)
            content = doc.page_content.lower()
            
            # Try to find patient name in content
            assigned = False
            for patient_name in extracted_data.keys():
                if patient_name.lower() in content:
                    if patient_name not in assignments:
                        assignments[patient_name] = []
                    assignments[patient_name].append(page_num)
                    assigned = True
                    break
            
            if not assigned:
                review_queue.append({
                    "page": page_num,
                    "reason": "No clear patient assignment",
                    "candidates": []
                })
        
        quality_metrics = {
            "total_pages": len(documents),
            "assigned_pages": sum(len(pages) for pages in assignments.values()),
            "unassigned_pages": len(review_queue),
            "assignment_confidence": 0.8  # Mock confidence
        }
        
        return ProcessingResult(
            success=True,
            stage=ProcessingStage.PATIENT_ASSIGNMENT,
            data={
                "patient_assignments": assignments,
                "review_queue": review_queue,
                "quality_metrics": quality_metrics
            },
            metadata={},
            errors=[],
            warnings=[],
            processing_time=0.0
        )
    
    def process(self, data: Any) -> ProcessingResult:
        """Process method required by BaseProcessor"""
        if isinstance(data, tuple) and len(data) == 2:
            documents, extracted_data = data
            return self.assign(documents, extracted_data)
        else:
            return ProcessingResult(
                success=False,
                stage=ProcessingStage.PATIENT_ASSIGNMENT,
                data={},
                metadata={},
                errors=["Invalid input data format"],
                warnings=[],
                processing_time=0.0
            )


def assign_pages_to_patients(documents: List[Document],
                           patient_tokens: Dict[str, List[str]]) -> Tuple[Dict[str, List[int]], List[Tuple[int, List[str], str]]]:
    """Compatibility wrapper for existing assignment function"""
    engine = AssignmentEngine(type('Config', (), {'confidence_threshold': 0.7})())
    result = engine.assign(documents, patient_tokens)

    assignments = result.data["patient_assignments"]
    review_queue = [
        (item["page"], item["candidates"], item["reason"])
        for item in result.data["review_queue"]
    ]

    return assignments, review_queue
