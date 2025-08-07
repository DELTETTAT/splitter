import re
import os
import gc
import cv2
import json
import time
import fitz
import logging
import pytesseract
import numpy as np

from os import path
from PIL import Image
from collections import Counter
from langchain.schema import Document
from typing import List, Optional, Any

from core.base import BaseExtractor, ProcessingResult, ProcessingStage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deskew_image(cv_img):
    """Correct image skew for better OCR"""
    coords = np.column_stack(np.where(cv_img > 0))
    if len(coords) < 5:
        return cv_img  # nothing to deskew
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = cv_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def clean_image_for_ocr(pil_img):
    """Clean and enhance image for better OCR results"""
    cv_img = np.array(pil_img)
    if cv_img.mean() > 200:
        return pil_img
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    # Bilateral filtering to smooth noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold to handle uneven backgrounds
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

    # Morphological opening to remove small noise
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Deskew to straighten text lines
    deskewed = deskew_image(opened)

    # Scale up if small for better OCR
    if cv_img.shape[0] < 1000:
        scaled = cv2.resize(deskewed, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    else:
        scaled = deskewed

    return Image.fromarray(scaled)

def needs_ocr(page):
    """Determine if page needs OCR processing"""
    text = page.get_text().strip()
    if len(text) > 100 and re.search(r'\w+\s+\w+', text):
        return False
    if len(page.get_images()) > 3 and len(text) < 20:
        return False
    return True

class PDFExtractor(BaseExtractor):
    """Enhanced PDF text extractor with OCR and progress tracking"""

    def __init__(self, config, progress_callback=None):
        super().__init__(config)
        self.ocr_confidence_threshold = getattr(config, 'ocr_confidence_threshold', 70)
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)

    def extract(self, documents: List[Document]) -> ProcessingResult:
        """Extract text from PDF file path"""
        pass  # This will be called by text_extractor

    def process(self, data: Any) -> ProcessingResult:
        """Process method required by BaseProcessor"""
        if isinstance(data, str):
            # If data is a file path, extract text
            documents = self.text_extractor(data)
            return ProcessingResult(
                success=True,
                stage=ProcessingStage.TEXT_EXTRACTION,
                data={"documents": documents},
                metadata={},
                errors=[],
                warnings=[],
                processing_time=0.0
            )
        else:
            return ProcessingResult(
                success=False,
                stage=ProcessingStage.TEXT_EXTRACTION,
                data={},
                metadata={},
                errors=["Invalid input data type"],
                warnings=[],
                processing_time=0.0
            )

    def _process_single_page(self, page_index, pdf_path):
        """Process a single page without multiprocessing to avoid deadlock"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_index]
            text = page.get_text().strip()
            metadata = {
                "page": page_index + 1,
                "source_pdf": str(pdf_path),
                "ocr": False,
                "header_footer_removed": False
            }

            # If we have sufficient text, use it directly
            if text and not needs_ocr(page):
                doc.close()
                gc.collect()
                return Document(page_content=text, metadata=metadata)

            # OCR processing for pages with insufficient text
            try:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                processed_image = clean_image_for_ocr(img)

                # Use OCR with timeout and error handling
                ocr_text = pytesseract.image_to_string(processed_image, timeout=30)
                metadata.update({"ocr": True})

                doc.close()
                gc.collect()
                return Document(page_content=ocr_text, metadata=metadata)

            except Exception as ocr_error:
                self.logger.warning(f"OCR failed for page {page_index + 1}: {ocr_error}")
                # Return whatever text we could extract
                metadata.update({"ocr_failed": True, "error": str(ocr_error)})
                doc.close()
                gc.collect()
                return Document(page_content=text, metadata=metadata)

        except Exception as e:
            self.logger.error(f"Error processing page {page_index + 1}: {e}")
            return Document(
                page_content="",
                metadata={
                    "page": page_index + 1,
                    "source_pdf": str(pdf_path),
                    "error": str(e)
                }
            )

    def text_extractor(self, pdf_path: str) -> List[Document]:
        """Main extraction method with sequential processing to avoid deadlock"""
        start_time = time.time()

        if not os.path.exists(pdf_path):
            self.logger.error(f"File not found: {pdf_path}")
            return []

        self._update_progress(ProcessingStage.INITIALIZATION, 0.0, f"Opening PDF: {pdf_path}")

        try:
            # Get total pages
            base_doc = fitz.open(pdf_path)
            total_pages = len(base_doc)
            self.logger.info(f"Total pages: {total_pages}")
            base_doc.close()
        except Exception as e:
            self.logger.error(f"Error opening PDF {pdf_path}: {e}")
            return []

        self._update_progress(ProcessingStage.TEXT_EXTRACTION, 0.1, 
                            f"Starting extraction of {total_pages} pages")

        documents = []

        # Process pages sequentially to avoid multiprocessing deadlock
        for i in range(total_pages):
            try:
                doc = self._process_single_page(i, pdf_path)
                documents.append(doc)

                progress = 0.1 + ((i + 1) / total_pages) * 0.8
                self._update_progress(ProcessingStage.TEXT_EXTRACTION, progress,
                                    f"Processed page {i + 1}/{total_pages}")

            except Exception as e:
                self.logger.error(f"Error processing page {i + 1}: {e}")
                documents.append(Document(
                    page_content="",
                    metadata={
                        "page": i + 1,
                        "source_pdf": str(pdf_path),
                        "error": str(e)
                    }
                ))

        elapsed = time.time() - start_time
        self.logger.info(f"Extraction completed in {elapsed:.1f}s")

        self._update_progress(ProcessingStage.TEXT_EXTRACTION, 1.0, 
                            f"Extraction complete: {total_pages} pages in {elapsed:.1f}s")

        return documents

    def _update_progress(self, stage: ProcessingStage, progress: float, message: str):
        """Update progress if callback is available"""
        if self.progress_callback:
            self.progress_callback.update(stage, progress, message)