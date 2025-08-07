import tkinter as tk
from tkinter import filedialog, messagebox
from extractor import PDFExtractor
from extractor import NameSeedExtractor
from assigner import NamePageAssigner

class ReviewUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Splitter Review UI")
        self.geometry("800x600")
        self.extractor = PDFExtractor()
        self.seeder = NameSeedExtractor()
        self.assigner = None
        self.docs = []
        self.page_map = {}
        self.setup_widgets()

    def setup_widgets(self):
        btn_load = tk.Button(self, text="Load PDF", command=self.load_pdf)
        btn_load.pack(pady=10)
        self.listbox = tk.Listbox(self, width=100)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        btn_split = tk.Button(self, text="Split PDFs", command=self.split_pdfs)
        btn_split.pack(pady=10)

    def load_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not path: return
        self.docs = self.extractor.extract_pages(path)
        seed_map = self.seeder.extract(self.docs)
        self.assigner = NamePageAssigner(seed_map)
        self.page_map = {i: next(iter(names)) for i, names in seed_map.items()}
        self.page_map = self.assigner.assign(self.docs, self.page_map)
        self.refresh_list()

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i, doc in enumerate(self.docs):
            name = self.page_map.get(i, "<Unassigned>")
            self.listbox.insert(tk.END, f"Page {i+1}: {name}")

    def split_pdfs(self):
        from splitter import PDFSplitter
        splitter = PDFSplitter()
        try:
            splitter.split(self.docs, self.page_map)
            messagebox.showinfo("Success", "PDFs split successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = ReviewUI()
    app.mainloop()