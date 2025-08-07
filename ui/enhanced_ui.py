
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

from config import CONFIG
from extractors.pdf_extractor import PDFExtractor
from extractors.name_extractor import NameExtractor
from processors.assignment_engine import AssignmentEngine
from ai.local_engine import LocalAIEngine
from workflows.langgraph_processor import LangGraphProcessor
from utils.progress_tracker import ProgressTracker, ProgressCallback
from utils.splitter import PDFSplitter
from core.base import ProcessingStage


class EnhancedReviewUI:
    """Enhanced GUI interface with comprehensive progress tracking"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🏥 Enhanced Medical Record PDF Splitter")
        self.root.geometry("1000x700")
        
        # Data storage
        self.documents = []
        self.current_pdf = None
        self.processing_results = None
        self.assignments = {}
        self.review_queue = []
        
        # Progress tracking
        self.progress_tracker = ProgressTracker()
        self.progress_callback = GUIProgressCallback(self)
        self.progress_tracker.add_callback("gui", self.progress_callback)
        
        # UI state
        self.processing_active = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI components"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padding="10")
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill="x", pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="🏥 Enhanced Medical Record PDF Splitter", 
                               font=("Arial", 16, "bold"))
        title_label.pack()
        
        config_label = ttk.Label(header_frame, 
                                text=f"Device: {CONFIG.device} | Mode: {CONFIG.speed_vs_accuracy} | Confidence: {CONFIG.confidence_threshold:.2f}")
        config_label.pack()
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="📄 PDF Selection", padding="10")
        file_frame.pack(fill="x", pady=(0, 10))
        
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_var, width=60)
        file_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_btn.pack(side="right")
        
        # Processing controls
        control_frame = ttk.LabelFrame(main_frame, text="🔧 Processing Controls", padding="10")
        control_frame.pack(fill="x", pady=(0, 10))
        
        # Processing options
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill="x", pady=(0, 10))
        
        self.use_langgraph_var = tk.BooleanVar(value=CONFIG.use_langgraph)
        langgraph_check = ttk.Checkbutton(options_frame, text="Use AI Processing (LangGraph)", 
                                         variable=self.use_langgraph_var)
        langgraph_check.pack(side="left")
        
        # Processing buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x")
        
        self.process_btn = ttk.Button(button_frame, text="🤖 Process with AI", 
                                     command=self.start_processing, style="Accent.TButton")
        self.process_btn.pack(side="left", padx=(0, 5))
        
        self.traditional_btn = ttk.Button(button_frame, text="🔍 Traditional Processing", 
                                         command=self.start_traditional_processing)
        self.traditional_btn.pack(side="left", padx=(0, 5))
        
        self.split_btn = ttk.Button(button_frame, text="✂️ Split PDFs", 
                                   command=self.split_pdfs, state="disabled")
        self.split_btn.pack(side="right")
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="📊 Processing Progress", padding="10")
        progress_frame.pack(fill="x", pady=(0, 10))
        
        # Overall progress
        ttk.Label(progress_frame, text="Overall Progress:").pack(anchor="w")
        self.overall_progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.overall_progress.pack(fill="x", pady=(0, 5))
        
        # Stage progress
        ttk.Label(progress_frame, text="Current Stage:").pack(anchor="w")
        self.stage_progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.stage_progress.pack(fill="x", pady=(0, 5))
        
        # Status labels
        self.status_label = ttk.Label(progress_frame, text="Ready to process...")
        self.status_label.pack(anchor="w")
        
        self.stage_label = ttk.Label(progress_frame, text="")
        self.stage_label.pack(anchor="w")
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="📋 Results", padding="10")
        results_frame.pack(fill="both", expand=True)
        
        # Results notebook
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill="both", expand=True)
        
        # Assignments tab
        assignments_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(assignments_frame, text="👥 Patient Assignments")
        
        self.assignments_text = scrolledtext.ScrolledText(assignments_frame, wrap="word")
        self.assignments_text.pack(fill="both", expand=True)
        
        # Review queue tab
        review_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(review_frame, text="⚠️ Review Queue")
        
        self.review_text = scrolledtext.ScrolledText(review_frame, wrap="word")
        self.review_text.pack(fill="both", expand=True)
        
        # Metrics tab
        metrics_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(metrics_frame, text="📈 Quality Metrics")
        
        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, wrap="word")
        self.metrics_text.pack(fill="both", expand=True)
    
    def browse_file(self):
        """Browse for PDF file"""
        filename = filedialog.askopenfilename(
            title="Select PDF file",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.file_var.set(filename)
            self.current_pdf = filename
    
    def start_processing(self):
        """Start AI processing in background thread"""
        if not self.current_pdf or not Path(self.current_pdf).exists():
            messagebox.showerror("Error", "Please select a valid PDF file")
            return
        
        if self.processing_active:
            return
        
        self.processing_active = True
        self.process_btn.config(state="disabled")
        self.traditional_btn.config(state="disabled")
        
        # Start processing in background thread
        thread = threading.Thread(target=self._process_documents, args=(True,))
        thread.daemon = True
        thread.start()
    
    def start_traditional_processing(self):
        """Start traditional processing in background thread"""
        if not self.current_pdf or not Path(self.current_pdf).exists():
            messagebox.showerror("Error", "Please select a valid PDF file")
            return
        
        if self.processing_active:
            return
        
        self.processing_active = True
        self.process_btn.config(state="disabled")
        self.traditional_btn.config(state="disabled")
        
        # Start processing in background thread
        thread = threading.Thread(target=self._process_documents, args=(False,))
        thread.daemon = True
        thread.start()
    
    def _process_documents(self, use_ai: bool):
        """Process documents with progress tracking"""
        try:
            # Extract documents
            extractor = PDFExtractor(CONFIG, self.progress_tracker)
            self.documents = extractor.text_extractor(self.current_pdf)
            
            if use_ai and self.use_langgraph_var.get():
                # AI processing with LangGraph
                processor = LangGraphProcessor(CONFIG, self.progress_tracker)
                result_state = processor.process_documents(self.documents)
                
                self.assignments = result_state.get("patient_assignments", {})
                self.review_queue = result_state.get("review_queue", [])
                quality_metrics = result_state.get("metadata", {}).get("quality_metrics", {})
                
            else:
                # Traditional processing
                name_extractor = NameExtractor(CONFIG, self.progress_tracker)
                name_result = name_extractor.extract(self.documents)
                consolidated = name_extractor.get_consolidated_name_tokens(min_pages=1)
                
                assignment_engine = AssignmentEngine(CONFIG, self.progress_tracker)
                assignment_result = assignment_engine.assign(self.documents, consolidated)
                
                self.assignments = assignment_result.data["patient_assignments"]
                self.review_queue = assignment_result.data["review_queue"]
                quality_metrics = assignment_result.data["quality_metrics"]
            
            # Update UI with results
            self.root.after(0, self._update_results_ui, quality_metrics)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Processing Error", str(e)))
        finally:
            self.root.after(0, self._processing_complete)
    
    def _update_results_ui(self, quality_metrics: Dict[str, Any]):
        """Update results UI components"""
        # Update assignments
        self.assignments_text.delete(1.0, tk.END)
        assignments_content = f"👥 Patient Assignments ({len(self.assignments)} patients):\n\n"
        for patient, pages in self.assignments.items():
            assignments_content += f"📋 {patient}: {len(pages)} pages ({sorted(pages)})\n"
        self.assignments_text.insert(1.0, assignments_content)
        
        # Update review queue
        self.review_text.delete(1.0, tk.END)
        review_content = f"⚠️ Review Queue ({len(self.review_queue)} pages):\n\n"
        for item in self.review_queue:
            page = item.get("page", "")
            reason = item.get("reason", "")
            review_content += f"📄 Page {page}: {reason}\n"
        self.review_text.insert(1.0, review_content)
        
        # Update metrics
        self.metrics_text.delete(1.0, tk.END)
        metrics_content = "📈 Quality Metrics:\n\n"
        for key, value in quality_metrics.items():
            metrics_content += f"{key.replace('_', ' ').title()}: {value}\n"
        self.metrics_text.insert(1.0, metrics_content)
        
        # Enable split button
        self.split_btn.config(state="normal")
    
    def _processing_complete(self):
        """Reset UI after processing completion"""
        self.processing_active = False
        self.process_btn.config(state="normal")
        self.traditional_btn.config(state="normal")
        self.status_label.config(text="Processing complete!")
        self.stage_label.config(text="")
        self.overall_progress["value"] = 100
        self.stage_progress["value"] = 100
    
    def split_pdfs(self):
        """Split PDFs based on assignments"""
        if not self.assignments:
            messagebox.showwarning("Warning", "No assignments available for splitting")
            return
        
        output_dir = filedialog.askdirectory(title="Select output directory")
        if not output_dir:
            return
        
        try:
            splitter = PDFSplitter()
            
            # Convert assignments to page-based format
            page_assignments = {}
            for patient, pages in self.assignments.items():
                for page in pages:
                    page_assignments[page - 1] = patient  # Convert to 0-based
            
            splitter.split(self.documents, page_assignments, output_dir)
            messagebox.showinfo("Success", f"PDFs split successfully!\nOutput: {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Split Error", f"Failed to split PDFs: {e}")
    
    def update_progress(self, ui_data: Dict[str, Any]):
        """Update progress UI components"""
        if not self.processing_active:
            return
        
        current_stage = ui_data.get("current_stage", "")
        stage_progress = ui_data.get("stage_progress", 0) * 100
        stage_message = ui_data.get("stage_message", "")
        
        # Update labels
        self.status_label.config(text=f"Stage: {current_stage.replace('_', ' ').title()}")
        self.stage_label.config(text=stage_message)
        
        # Update progress bars
        self.stage_progress["value"] = stage_progress
        
        # Calculate overall progress based on stage
        stage_weights = {
            "initialization": 10,
            "text_extraction": 30,
            "name_extraction": 20,
            "classification": 15,
            "entity_extraction": 10,
            "patient_assignment": 10,
            "validation": 3,
            "quality_check": 1,
            "summary_generation": 1
        }
        
        overall_progress = 0
        for stage, weight in stage_weights.items():
            if stage in current_stage.lower():
                overall_progress += weight * (stage_progress / 100)
                break
        
        self.overall_progress["value"] = min(overall_progress, 95)  # Cap at 95% until complete
    
    def mainloop(self):
        """Start the GUI main loop"""
        self.root.mainloop()


class GUIProgressCallback:
    """Progress callback for GUI updates"""
    
    def __init__(self, gui_instance):
        self.gui = gui_instance
    
    def __call__(self, stage, progress, message, metadata=None):
        """Update GUI progress"""
        ui_data = {
            "current_stage": stage.value if hasattr(stage, 'value') else str(stage),
            "stage_progress": progress,
            "stage_message": message,
            "metadata": metadata or {}
        }
        
        # Schedule UI update on main thread
        self.gui.root.after(0, self.gui.update_progress, ui_data)


if __name__ == "__main__":
    app = EnhancedReviewUI()
    app.mainloop()
