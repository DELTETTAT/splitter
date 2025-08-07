from pathlib import Path
import pikepdf
from langchain.schema import Document
from typing import Dict, List

class PDFSplitter:
    def split(self, docs: List[Document], assignments: Dict[int, str], output_dir: str = "output"):
        """
        Group pages by patient name and write one PDF per patient.
        """
        grouped: Dict[str, List[int]] = {}
        for idx, name in assignments.items():
            grouped.setdefault(name, []).append(idx)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # original PDF path is in metadata if needed
        for patient, pages in grouped.items():
            out_path = Path(output_dir) / f"{patient.replace(' ', '_')}.pdf"
            with pikepdf.Pdf.new() as pdf_out:
                # for each page index, import the page
                original = docs[0].metadata.get('source_pdf')
                src = pikepdf.Pdf.open(original)
                for i in sorted(pages):
                    pdf_out.pages.append(src.pages[i])
                pdf_out.save(out_path)