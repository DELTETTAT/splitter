
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from typing import Dict, Any, List
from extractor import PDFExtractor
from local_ai_engine import AI_ENGINE
from langgraph_processor import LANGGRAPH_PROCESSOR
from config import CONFIG
import threading
import logging

logger = logging.getLogger(__name__)

class EnhancedReviewUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Enhanced Medical Record Splitter")
        self.geometry("1200x800")
        
        # Processing state
        self.documents = []
        self.processing_state = None
        self.current_pdf_path = None
        
        self.setup_ui()
        self.update_status("Ready")
    
    def setup_ui(self):
        """Setup the enhanced UI with configuration panels"""
        
        # Create main frames
        self.create_menu()
        self.create_config_panel()
        self.create_processing_panel()
        self.create_results_panel()
        self.create_status_bar()
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load PDF", command=self.load_pdf)
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        config_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Configuration", menu=config_menu)
        config_menu.add_command(label="Save Config", command=self.save_config)
        config_menu.add_command(label="Load Config", command=self.load_config)
    
    def create_config_panel(self):
        """Create configuration panel"""
        config_frame = ttk.LabelFrame(self, text="Processing Configuration")
        config_frame.pack(fill="x", padx=5, pady=5)
        
        # Speed vs Accuracy
        ttk.Label(config_frame, text="Speed vs Accuracy:").grid(row=0, column=0, sticky="w", padx=5)
        self.speed_var = tk.StringVar(value=CONFIG.speed_vs_accuracy)
        speed_combo = ttk.Combobox(config_frame, textvariable=self.speed_var, 
                                  values=["fast", "balanced", "accurate"], state="readonly")
        speed_combo.grid(row=0, column=1, padx=5, pady=2)
        speed_combo.bind("<<ComboboxSelected>>", self.update_config)
        
        # Confidence Threshold
        ttk.Label(config_frame, text="Confidence Threshold:").grid(row=0, column=2, sticky="w", padx=5)
        self.confidence_var = tk.DoubleVar(value=CONFIG.confidence_threshold)
        confidence_scale = ttk.Scale(config_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_var, orient="horizontal", length=100)
        confidence_scale.grid(row=0, column=3, padx=5, pady=2)
        confidence_scale.bind("<Motion>", self.update_config)
        
        self.confidence_label = ttk.Label(config_frame, text=f"{CONFIG.confidence_threshold:.2f}")
        self.confidence_label.grid(row=0, column=4, padx=5)
        
        # Batch Size
        ttk.Label(config_frame, text="Batch Size:").grid(row=1, column=0, sticky="w", padx=5)
        self.batch_var = tk.IntVar(value=CONFIG.batch_size)
        batch_spin = ttk.Spinbox(config_frame, from_=1, to=16, textvariable=self.batch_var, width=5)
        batch_spin.grid(row=1, column=1, padx=5, pady=2)
        batch_spin.bind("<KeyRelease>", self.update_config)
        
        # Enable LangGraph
        self.langgraph_var = tk.BooleanVar(value=CONFIG.use_langgraph)
        ttk.Checkbutton(config_frame, text="Use LangGraph", 
                       variable=self.langgraph_var, command=self.update_config).grid(row=1, column=2, padx=5)
        
        # Enable GraphRAG
        self.graphrag_var = tk.BooleanVar(value=CONFIG.graph_rag_enabled)
        ttk.Checkbutton(config_frame, text="Enable GraphRAG", 
                       variable=self.graphrag_var, command=self.update_config).grid(row=1, column=3, padx=5)
        
        # Device Info
        device_info = f"Device: {CONFIG.device} | GPU: {'Available' if CONFIG.use_gpu else 'Not Available'}"
        ttk.Label(config_frame, text=device_info, foreground="blue").grid(row=2, column=0, columnspan=5, sticky="w", padx=5, pady=5)
    
    def create_processing_panel(self):
        """Create processing control panel"""
        process_frame = ttk.LabelFrame(self, text="Processing Controls")
        process_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(process_frame, text="Load PDF", command=self.load_pdf).pack(side="left", padx=5, pady=5)
        ttk.Button(process_frame, text="Process with LangGraph", command=self.process_with_langgraph).pack(side="left", padx=5, pady=5)
        ttk.Button(process_frame, text="Split PDFs", command=self.split_pdfs).pack(side="left", padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(process_frame, variable=self.progress_var, length=200)
        self.progress_bar.pack(side="right", padx=5, pady=5)
    
    def create_results_panel(self):
        """Create results display panel"""
        results_frame = ttk.LabelFrame(self, text="Processing Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create notebook for different result views
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Assignments tab
        self.assignments_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.assignments_frame, text="Assignments")
        
        self.assignments_tree = ttk.Treeview(self.assignments_frame, columns=("Pages", "Confidence"), show="tree headings")
        self.assignments_tree.heading("#0", text="Patient Name")
        self.assignments_tree.heading("Pages", text="Pages")
        self.assignments_tree.heading("Confidence", text="Avg Confidence")
        self.assignments_tree.pack(fill="both", expand=True)
        
        # Review queue tab
        self.review_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.review_frame, text="Review Queue")
        
        self.review_tree = ttk.Treeview(self.review_frame, columns=("Candidates", "Reason", "Confidence"), show="tree headings")
        self.review_tree.heading("#0", text="Page")
        self.review_tree.heading("Candidates", text="Candidates")
        self.review_tree.heading("Reason", text="Reason")
        self.review_tree.heading("Confidence", text="Confidence")
        self.review_tree.pack(fill="both", expand=True)
        
        # Quality metrics tab
        self.metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_frame, text="Quality Metrics")
        
        self.metrics_text = tk.Text(self.metrics_frame, height=10)
        self.metrics_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Summaries tab
        self.summaries_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summaries_frame, text="Patient Summaries")
        
        self.summaries_text = tk.Text(self.summaries_frame, height=10)
        self.summaries_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken")
        status_bar.pack(side="bottom", fill="x")
    
    def update_status(self, message: str):
        """Update status bar message"""
        self.status_var.set(message)
        self.update_idletasks()
    
    def update_config(self, event=None):
        """Update configuration based on UI changes"""
        CONFIG.speed_vs_accuracy = self.speed_var.get()
        CONFIG.confidence_threshold = self.confidence_var.get()
        CONFIG.batch_size = self.batch_var.get()
        CONFIG.use_langgraph = self.langgraph_var.get()
        CONFIG.graph_rag_enabled = self.graphrag_var.get()
        
        # Update confidence label
        self.confidence_label.config(text=f"{CONFIG.confidence_threshold:.2f}")
        
        logger.info(f"Configuration updated: {CONFIG.__dict__}")
    
    def load_pdf(self):
        """Load PDF file"""
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if not file_path:
            return
        
        self.current_pdf_path = file_path
        self.update_status("Loading PDF...")
        
        # Start processing in background thread
        threading.Thread(target=self._load_pdf_thread, args=(file_path,), daemon=True).start()
    
    def _load_pdf_thread(self, file_path: str):
        """Load PDF in background thread"""
        try:
            extractor = PDFExtractor(
                ocr_confidence_threshold=CONFIG.ocr_confidence_threshold,
                num_workers=CONFIG.max_workers
            )
            
            self.documents = extractor.text_extractor(file_path)
            
            self.after(0, lambda: self.update_status(f"Loaded {len(self.documents)} pages"))
            self.after(0, lambda: messagebox.showinfo("Success", f"Loaded {len(self.documents)} pages"))
            
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to load PDF: {str(e)}"))
    
    def process_with_langgraph(self):
        """Process documents using LangGraph"""
        if not self.documents:
            messagebox.showwarning("Warning", "Please load a PDF first")
            return
        
        self.update_status("Processing with LangGraph...")
        self.progress_var.set(0)
        
        # Start processing in background thread
        threading.Thread(target=self._process_langgraph_thread, daemon=True).start()
    
    def _process_langgraph_thread(self):
        """Process documents with LangGraph in background thread"""
        try:
            if CONFIG.use_langgraph:
                self.processing_state = LANGGRAPH_PROCESSOR.process_documents(self.documents)
            else:
                # Fallback to traditional processing
                from assigner import assign_pages_to_patients
                from name_extractor import NameExtractor
                
                name_extractor = NameExtractor()
                name_extractor.extract(self.documents)
                consolidated = name_extractor.get_consolidated_name_tokens(min_pages=1)
                
                assignments, review = assign_pages_to_patients(self.documents, consolidated)
                
                self.processing_state = {
                    "patient_assignments": assignments,
                    "review_queue": [
                        {"page": page, "candidates": candidates, "reason": reason}
                        for page, candidates, reason in review
                    ],
                    "confidence_scores": {},
                    "metadata": {"quality_metrics": {}}
                }
            
            self.after(0, self._update_results_display)
            self.after(0, lambda: self.update_status("Processing complete"))
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            self.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
    
    def _update_results_display(self):
        """Update the results display with processing results"""
        if not self.processing_state:
            return
        
        # Clear existing results
        for item in self.assignments_tree.get_children():
            self.assignments_tree.delete(item)
        
        for item in self.review_tree.get_children():
            self.review_tree.delete(item)
        
        # Update assignments
        assignments = self.processing_state.get("patient_assignments", {})
        confidence_scores = self.processing_state.get("confidence_scores", {})
        
        for patient, pages in assignments.items():
            page_list = ", ".join(map(str, sorted(pages)))
            avg_confidence = sum(confidence_scores.get(p, 0.5) for p in pages) / len(pages) if pages else 0
            
            self.assignments_tree.insert("", "end", text=patient, values=(page_list, f"{avg_confidence:.2f}"))
        
        # Update review queue
        review_queue = self.processing_state.get("review_queue", [])
        for item in review_queue:
            page = item.get("page", "")
            candidates = item.get("candidates", [])
            reason = item.get("reason", "")
            confidence = item.get("confidence", 0.0)
            
            candidates_str = ", ".join(candidates) if isinstance(candidates, list) else str(candidates)
            self.review_tree.insert("", "end", text=str(page), 
                                  values=(candidates_str, reason, f"{confidence:.2f}"))
        
        # Update quality metrics
        metrics = self.processing_state.get("metadata", {}).get("quality_metrics", {})
        metrics_text = "Quality Metrics:\n\n"
        for key, value in metrics.items():
            metrics_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics_text)
        
        # Update summaries
        summaries = self.processing_state.get("metadata", {}).get("patient_summaries", {})
        summaries_text = "Patient Summaries:\n\n"
        for patient, summary in summaries.items():
            summaries_text += f"{patient}:\n{summary}\n\n"
        
        self.summaries_text.delete(1.0, tk.END)
        self.summaries_text.insert(1.0, summaries_text)
    
    def split_pdfs(self):
        """Split PDFs based on assignments"""
        if not self.processing_state or not self.current_pdf_path:
            messagebox.showwarning("Warning", "Please process documents first")
            return
        
        try:
            from splitter import PDFSplitter
            splitter = PDFSplitter()
            
            assignments = self.processing_state.get("patient_assignments", {})
            # Convert to the format expected by splitter
            page_assignments = {}
            for patient, pages in assignments.items():
                for page in pages:
                    page_assignments[page - 1] = patient  # Convert to 0-based indexing
            
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if output_dir:
                splitter.split(self.documents, page_assignments, output_dir)
                messagebox.showinfo("Success", f"PDFs split successfully in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error splitting PDFs: {e}")
            messagebox.showerror("Error", f"Failed to split PDFs: {str(e)}")
    
    def save_config(self):
        """Save current configuration"""
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                config_dict = {
                    "speed_vs_accuracy": CONFIG.speed_vs_accuracy,
                    "confidence_threshold": CONFIG.confidence_threshold,
                    "batch_size": CONFIG.batch_size,
                    "use_langgraph": CONFIG.use_langgraph,
                    "graph_rag_enabled": CONFIG.graph_rag_enabled,
                    "ocr_confidence_threshold": CONFIG.ocr_confidence_threshold,
                    "max_workers": CONFIG.max_workers
                }
                
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                messagebox.showinfo("Success", "Configuration saved successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def load_config(self):
        """Load configuration from file"""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config_dict = json.load(f)
                
                CONFIG.update_from_dict(config_dict)
                
                # Update UI elements
                self.speed_var.set(CONFIG.speed_vs_accuracy)
                self.confidence_var.set(CONFIG.confidence_threshold)
                self.batch_var.set(CONFIG.batch_size)
                self.langgraph_var.set(CONFIG.use_langgraph)
                self.graphrag_var.set(CONFIG.graph_rag_enabled)
                
                self.update_config()
                messagebox.showinfo("Success", "Configuration loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
    
    def export_results(self):
        """Export processing results"""
        if not self.processing_state:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.processing_state, f, indent=2, default=str)
                
                messagebox.showinfo("Success", "Results exported successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = EnhancedReviewUI()
    app.mainloop()
