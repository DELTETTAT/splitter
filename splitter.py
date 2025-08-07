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

        # Get original PDF path from metadata
        original_pdf = None
        for doc in docs:
            if doc.metadata.get('source_pdf'):
                original_pdf = doc.metadata['source_pdf']
                break

        if not original_pdf:
            raise ValueError("No source PDF path found in document metadata")

        # Split PDFs by patient
        for patient, pages in grouped.items():
            safe_name = "".join(c for c in patient if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            out_path = Path(output_dir) / f"{safe_name}.pdf"

            try:
                with pikepdf.Pdf.new() as pdf_out:
                    src = pikepdf.Pdf.open(original_pdf)
                    for i in sorted(pages):
                        if i < len(src.pages):
                            pdf_out.pages.append(src.pages[i])
                    pdf_out.save(out_path)
                    src.close()
                print(f"Created: {out_path}")
            except Exception as e:
                print(f"Error creating PDF for {patient}: {e}")