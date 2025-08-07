import os
import json
import logging
from extractor import PDFExtractor
from collections import defaultdict
from typing import List, Tuple, Dict
from langchain.schema import Document
from name_extractor import NameExtractor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PageAssigner:
    def __init__(self, patient_tokens: Dict[str, List[str]]):
        self.patient_tokens = patient_tokens  # {full_name: [tokens]}

    def assign(self, docs: List[Document]) -> Tuple[Dict[str, List[int]], List[Tuple[int, List[str], str]]]:
        # Collect page matches and all pages
        page_matches: Dict[int, List[str]] = defaultdict(list)
        all_pages: List[int] = []

        # Seed from metadata
        for doc in docs:
            page = doc.metadata.get("page")
            all_pages.append(page)
            meta_name = doc.metadata.get("patient_name")
            if meta_name and meta_name in self.patient_tokens:
                page_matches[page].append(meta_name)

        # Token-based matching across text
        for doc in docs:
            page = doc.metadata.get("page")
            text_lower = (doc.page_content or "").lower()
            for name, tokens in self.patient_tokens.items():
                if any(token.lower() in text_lower for token in tokens):
                    if name not in page_matches[page]:
                        page_matches[page].append(name)

        # Build name-to-pages map including all valid matches
        name_to_pages: Dict[str, List[int]] = defaultdict(list)
        for page, matches in page_matches.items():
            for name in matches:
                if name in self.patient_tokens:
                    name_to_pages[name].append(page)

        # Initialize clusters
        name_to_canonical = {name: name for name in name_to_pages}
        clusters: Dict[str, set] = {name: {name} for name in name_to_pages}

        # Merge clusters based on shared tokens and pages
        names = list(name_to_pages.keys())
        for i, name1 in enumerate(names):
            canonical1 = name_to_canonical[name1]
            tokens1 = set(self.patient_tokens.get(name1, name1.split()))
            for j in range(i + 1, len(names)):
                name2 = names[j]
                canonical2 = name_to_canonical[name2]
                if canonical1 == canonical2:
                    continue
                tokens2 = set(self.patient_tokens.get(name2, name2.split()))
                # require at least one shared token
                if not tokens1 & tokens2:
                    continue
                pages1 = set(name_to_pages[name1])
                pages2 = set(name_to_pages[name2])
                # require at least one shared page
                if pages1 & pages2:
                    # merge cluster2 into cluster1
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

        # Map each page to its possible canonicals
        page_to_canonicals: Dict[int, set] = defaultdict(set)
        for canonical, pages in merged.items():
            for page in pages:
                page_to_canonicals[page].add(canonical)

        # Final assignments and review list
        assignments: Dict[str, List[int]] = defaultdict(list)
        review: List[Tuple[int, List[str], str]] = []
        assigned_pages: set = set()

        for page in all_pages:
            canonicals = page_to_canonicals.get(page, set())
            if len(canonicals) == 1:
                # unambiguous assignment
                name = next(iter(canonicals))
                assignments[name].append(page)
                assigned_pages.add(page)
            elif canonicals:
                # ambiguous: multiple clusters matched
                review.append((page, list(canonicals), "Multiple clusters matched"))
            else:
                # no match at all
                matches = page_matches.get(page, [])
                review.append((page, matches, "No cluster matched"))

        logger.info("Assigned pages to %d patients; %d pages need review", 
                    len(assignments), len(review))
        return dict(assignments), review


def assign_pages_to_patients(
    documents: List[Document],
    patient_tokens: Dict[str, List[str]]
) -> Tuple[Dict[str, List[int]], List[Tuple[int, List[str], str]]]:
    assigner = PageAssigner(patient_tokens)
    return assigner.assign(documents)

if __name__ == "__main__":
    pdf_path = r"C:\Users\pc\Desktop\medrec sample data\PDF sorting.pdf"
    logger.info(f"Running extractors and assigner on {pdf_path}")

    extractor = PDFExtractor()
    documents = extractor.text_extractor(pdf_path)

    name_extractor = NameExtractor()
    name_extractor.extract(documents)
    consolidated = name_extractor.get_consolidated_name_tokens(min_pages=1)

    assignments, review = assign_pages_to_patients(documents, consolidated)

    print("\n✅ Assignments:")
    for name, pages in assignments.items():
        print(f" - {name}: pages {pages}")

    print("\n🛠️ Pages needing review:")
    for page, matched, reason in review:
        print(f" - Page {page}: matched {matched} → {reason}")

    os.makedirs("results", exist_ok=True)
    with open("results/extractor_summary.json", "w") as f:
        json.dump({
            "Extracted Documents": len(documents),
            "Documents": [
                {
                    "Page": i + 1,
                    "Content": doc.page_content,
                    "Metadata": doc.metadata
                } for i, doc in enumerate(documents)
            ]
        }, f, indent=4)

    with open("results/name_extractor_summary.json", "w") as f:
        json.dump({
            "Consolidated Names": consolidated,
            "Total Unique Names": len(consolidated),
            "Full Name Registry": name_extractor.full_name_registry,
            "Name Counter": name_extractor.name_counter.most_common()
        }, f, indent=4)

    with open("results/assigner_summary.json", "w") as f:
        json.dump({
            "Assignments": assignments,
            "Review": review
        }, f, indent=4)
