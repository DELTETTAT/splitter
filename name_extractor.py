import os
import re
import json
import logging
from collections import Counter
from extractor import PDFExtractor
from langchain.schema import Document
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MetaNER pipeline (Facebook's bart-large NER) once
_meta_ner = None
def get_meta_ner_pipeline():
    global _meta_ner
    if _meta_ner is None:
        _meta_ner = pipeline(
            "ner",
            model="dslim/bert-large-NER",
            aggregation_strategy="simple",
            device=-1
        )
    return _meta_ner

class NameExtractor:
    """
    Extract patient/customer names via regex patterns + MetaNER fallback.
    """
    NAME_PATTERNS = [
        re.compile(r"(?:Customer Name|Patient Name)[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE),
        re.compile(r"Customer[:\-]?\s*([A-Z][A-Z\s]+),\s*([A-Z][A-Z\s]+)", re.IGNORECASE),
        re.compile(r"DELIVER TO\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE),
    ]

    NON_PATIENT_KEYWORDS = [
        "Medical", "Equipment", "System", "Service", "Group",
        "Confirmation", "Order", "Setup", "Phone", "Progress",
        "Encounter", "Digitally", "Signed", "Date", "Lite",
        "Respiratory", "Check", "Care", "Home", "Hospice",
        "Medication", "Oxygen", "Notes"
    ]

    def __init__(self):
        self.name_counter = Counter()
        self.full_name_registry = {}
        self.token_map = {}

    def extract_patient_name_from_structured(self, doc: Document):
        return doc.metadata.get("structured_data", {}).get("patient")

    def normalize_name(self, name: str) -> str:
        return " ".join(w.capitalize() for w in name.split()) if name else None

    def is_likely_patient_name(self, name: str) -> bool:
        if not name:
            return False
        for word in self.NON_PATIENT_KEYWORDS:
            if word.lower() in name.lower():
                return False
        # allow titlecase or ALLCAPS names, up to 4 tokens
        return bool(re.match(r'^([A-Z][a-z]*|[A-Z]+)(\s([A-Z][a-z]*|[A-Z]+)){0,3}$', name.strip()))

    def extract_patient_name_from_text(self, text: str) -> str:
        text = text or ""
        # 1) regex patterns
        for pattern in self.NAME_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    clean_name = " ".join(part.strip().title() for part in match if part.strip())
                else:
                    clean_name = self.normalize_name(match)
                if self.is_likely_patient_name(clean_name):
                    logger.info(f"[NameExtract] Accepted '{clean_name}' using regex {pattern.pattern}")
                    return clean_name
        # 2) MetaNER fallback
        ner = get_meta_ner_pipeline()
        entities = ner(text)
        # filter for PERSON
        persons = [e for e in entities if e.get('entity_group') == 'PER']
        if persons:
            # choose highest-score span
            best = max(persons, key=lambda e: e.get('score', 0))
            clean = best['word'].strip()
            clean = self.normalize_name(clean)
            if self.is_likely_patient_name(clean):
                logger.info(f"[NameExtract] Accepted '{clean}' via MetaNER span")
                return clean
        return None

    def extract(self, documents: list[Document]):
        for doc in documents:
            # 1) structured
            patient_name = self.extract_patient_name_from_structured(doc)
            if patient_name:
                patient_name = self.normalize_name(patient_name)
            else:
                # 2) text
                patient_name = self.extract_patient_name_from_text(doc.page_content)

            # register
            if patient_name:
                initials = "".join(w[0] for w in patient_name.split())
                current = self.full_name_registry.get(initials)
                if not current or len(patient_name) > len(current):
                    self.full_name_registry[initials] = patient_name
                self.name_counter[patient_name] += 1
                self.token_map[patient_name] = patient_name.split()

            doc.metadata["patient_name"] = patient_name
            logger.info(f"[NameExtract] Page {doc.metadata.get('page')} name: {patient_name}")
        return documents

    def get_consolidated_name_tokens(self, min_pages=1):
        consolidated = {}
        for initials, full_name in self.full_name_registry.items():
            count = sum(cnt for name, cnt in self.name_counter.items()
                        if ''.join(w[0] for w in name.split()) == initials)
            if count >= min_pages:
                consolidated[full_name] = self.token_map.get(full_name, full_name.split())
        return consolidated

if __name__ == "__main__":
    pdf_path = r"C:\Users\pc\Desktop\medrec sample data\PDF sorting.pdf"
    extractor = PDFExtractor()
    documents = extractor.text_extractor(pdf_path)

    name_extractor = NameExtractor()
    name_extractor.extract(documents)
    consolidated = name_extractor.get_consolidated_name_tokens(min_pages=1)

    # write extractor summary
    os.makedirs("results", exist_ok=True)
    with open("results/extractor_summary.json", "w") as f:
        json.dump({
            "Extracted Documents": len(documents),
            "Documents": [
                {"Page": i+1, "Content": doc.page_content, "Metadata": doc.metadata}
                for i, doc in enumerate(documents)
            ]
        }, f, indent=4)

    # write name extractor summary
    with open("results/name_extractor_summary.json", "w") as f:
        json.dump({
            "Consolidated Names": consolidated,
            "Total Unique Names": len(consolidated),
            "Full Name Registry": name_extractor.full_name_registry,
            "Name Counter": name_extractor.name_counter.most_common()
        }, f, indent=4)
