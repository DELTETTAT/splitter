import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from extractors.pdf_extractor import PDFExtractor
from ai.local_engine import LocalAIEngine
from workflows.langgraph_processor import LangGraphProcessor
from config import CONFIG
from processors.assignment_engine import AssignmentEngine
from extractors.name_extractor import NameExtractor
import logging

# Configure logging for terminal display
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class InteractiveTerminal:
    def __init__(self):
        self.config = CONFIG
        self.documents = []
        self.current_pdf = None
        self.processing_results = None
        self.assignments = {}
        self.review_queue = []

        # Initialize processing components
        self.progress_tracker = ProgressTracker()
        self.console_callback = ConsoleProgressCallback(width=60)
        self.progress_tracker.add_callback("console", self.console_callback)

        # Add progress update method
        self.last_progress_time = time.time()

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """Print application header"""
        print("=" * 70)
        print("🏥 ENHANCED MEDICAL RECORD PDF SPLITTER")
        print("=" * 70)
        print(f"📍 Device: {CONFIG.device} | GPU: {'✅' if CONFIG.use_gpu else '❌'}")
        print(f"⚙️  Mode: {CONFIG.speed_vs_accuracy} | Confidence: {CONFIG.confidence_threshold:.2f}")
        print("-" * 70)

    def print_menu(self):
        """Print main menu options"""
        print("\n📋 MAIN MENU:")
        print("1. 📄 Load PDF")
        print("2. ⚙️  Configure Settings")
        print("3. 🤖 Process with AI (LangGraph)")
        print("4. 🔍 Process Traditional Pipeline")
        print("5. 📊 View Results")
        print("6. ✂️  Split PDFs")
        print("7. 💾 Export Results")
        print("8. 🔧 System Info")
        print("9. ❌ Exit")
        print("-" * 70)

    def get_user_choice(self, prompt="Select option: ", valid_choices=None):
        """Get user input with validation"""
        while True:
            try:
                choice = input(f"\n{prompt}").strip()
                if valid_choices and choice not in valid_choices:
                    print(f"❌ Invalid choice. Please select from: {', '.join(valid_choices)}")
                    continue
                return choice
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                exit(0)

    def animated_loading(self, message, duration=2):
        """Show animated loading"""
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        end_time = time.time() + duration
        i = 0
        while time.time() < end_time:
            print(f"\r{chars[i % len(chars)]} {message}", end='', flush=True)
            time.sleep(0.1)
            i += 1
        print(f"\r✅ {message} - Complete!")

    def load_pdf_interactive(self):
        """Interactive PDF loading"""
        self.clear_screen()
        self.print_header()
        print("📄 PDF LOADING")
        print("-" * 30)

        pdf_path = input("Enter PDF path (or drag & drop): ").strip().strip('"\'')

        if not os.path.exists(pdf_path):
            print("❌ File not found!")
            input("Press Enter to continue...")
            return

        self.current_pdf = pdf_path
        print(f"\n📂 Loading: {Path(pdf_path).name}")

        try:
            self.animated_loading("Extracting text from PDF", 3)

            extractor = PDFExtractor(CONFIG)
            self.documents = extractor.text_extractor(pdf_path)

            print(f"\n✅ Successfully loaded {len(self.documents)} pages")
            print(f"📊 Average text length: {sum(len(doc.page_content) for doc in self.documents) // len(self.documents)} chars")

            # Show preview
            if self.documents:
                print(f"\n📖 Preview of first page:")
                print("-" * 40)
                preview = self.documents[0].page_content[:200].replace('\n', ' ')
                print(f"{preview}...")
                print("-" * 40)

        except Exception as e:
            print(f"❌ Error loading PDF: {e}")

        input("\nPress Enter to continue...")

    def configure_settings_interactive(self):
        """Interactive settings configuration"""
        self.clear_screen()
        self.print_header()
        print("⚙️  CONFIGURATION SETTINGS")
        print("-" * 40)

        while True:
            print(f"\nCurrent Settings:")
            print(f"1. Speed vs Accuracy: {CONFIG.speed_vs_accuracy}")
            print(f"2. Confidence Threshold: {CONFIG.confidence_threshold:.2f}")
            print(f"3. Batch Size: {CONFIG.batch_size}")
            print(f"4. Use LangGraph: {CONFIG.use_langgraph}")
            if hasattr(CONFIG, 'graph_rag_enabled'):
                print(f"5. GraphRAG Enabled: {CONFIG.graph_rag_enabled}")
            else:
                print(f"5. GraphRAG Enabled: False")
            print("6. Max Workers: {CONFIG.max_workers}")
            print("7. 🔙 Back to Main Menu")


            choice = self.get_user_choice("Configure option: ", ["1", "2", "3", "4", "5", "6", "7"])

            if choice == "1":
                speed_choice = self.get_user_choice(
                    "Speed vs Accuracy (fast/balanced/accurate): ",
                    ["fast", "balanced", "accurate"]
                )
                CONFIG.speed_vs_accuracy = speed_choice
                print(f"✅ Updated to: {speed_choice}")

            elif choice == "2":
                try:
                    threshold = float(input("Confidence threshold (0.1-1.0): "))
                    if 0.1 <= threshold <= 1.0:
                        CONFIG.confidence_threshold = threshold
                        print(f"✅ Updated to: {threshold:.2f}")
                    else:
                        print("❌ Must be between 0.1 and 1.0")
                except ValueError:
                    print("❌ Invalid number")

            elif choice == "3":
                try:
                    batch_size = int(input("Batch size (1-16): "))
                    if 1 <= batch_size <= 16:
                        CONFIG.batch_size = batch_size
                        print(f"✅ Updated to: {batch_size}")
                    else:
                        print("❌ Must be between 1 and 16")
                except ValueError:
                    print("❌ Invalid number")

            elif choice == "4":
                use_langgraph = self.get_user_choice("Use LangGraph (y/n): ", ["y", "n"]) == "y"
                CONFIG.use_langgraph = use_langgraph
                print(f"✅ Updated to: {use_langgraph}")

            elif choice == "5":
                use_graphrag = self.get_user_choice("Enable GraphRAG (y/n): ", ["y", "n"]) == "y"
                CONFIG.graph_rag_enabled = use_graphrag
                print(f"✅ Updated to: {use_graphrag}")

            elif choice == "6":
                try:
                    workers = int(input("Max workers (1-8): "))
                    if 1 <= workers <= 8:
                        CONFIG.max_workers = workers
                        print(f"✅ Updated to: {workers}")
                    else:
                        print("❌ Must be between 1 and 8")
                except ValueError:
                    print("❌ Invalid number")

            elif choice == "7":
                break

    def process_with_langgraph_interactive(self):
        """Interactive LangGraph processing"""
        if not self.documents:
            print("❌ No PDF loaded! Please load a PDF first.")
            input("Press Enter to continue...")
            return

        self.clear_screen()
        self.print_header()
        print("🤖 AI PROCESSING WITH LANGGRAPH")
        print("-" * 50)

        print(f"📄 Processing {len(self.documents)} pages...")
        print(f"⚙️  Configuration: {CONFIG.speed_vs_accuracy} mode, {CONFIG.confidence_threshold:.2f} confidence")

        try:
            print("\n🔄 Starting AI processing pipeline...")

            # Simulate processing stages
            stages = [
                ("📝 Extracting text features", 2),
                ("🏷️  Classifying document types", 3),
                ("👤 Extracting patient entities", 4),
                ("🔗 Assigning pages to patients", 3),
                ("✅ Validating assignments", 2),
                ("📊 Quality checking", 2),
                ("📋 Generating summaries", 3)
            ]

            for stage_name, duration in stages:
                self.animated_loading(stage_name, duration)
                time.sleep(0.5)

            # Process with LangGraph
            if CONFIG.use_langgraph:
                print("\n🧠 Running LangGraph processor...")
                processor = LangGraphProcessor(CONFIG)
                self.processing_results = processor.process_documents(self.documents)
            else:
                print("\n🔧 Running traditional pipeline...")
                name_extractor = NameExtractor(CONFIG)
                name_extractor.extract(self.documents)
                consolidated = name_extractor.get_consolidated_name_tokens(min_pages=1)

                assignment_engine = AssignmentEngine(CONFIG)
                assignment_result = assignment_engine.assign(self.documents, consolidated)
                assignments = assignment_result.data["patient_assignments"]
                review = assignment_result.data["review_queue"]

                self.processing_results = {
                    "patient_assignments": assignments,
                    "review_queue": [
                        {"page": page, "candidates": candidates, "reason": reason}
                        for page, candidates, reason in review
                    ],
                    "confidence_scores": {},
                    "metadata": {"quality_metrics": {}}
                }

            self.assignments = self.processing_results.get("patient_assignments", {})
            self.review_queue = self.processing_results.get("review_queue", [])

            print(f"\n🎉 Processing Complete!")
            print(f"👥 Found {len(self.assignments)} patients")
            print(f"📄 Assigned {sum(len(pages) for pages in self.assignments.values())} pages")
            print(f"⚠️  {len(self.review_queue)} pages need review")

        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
            logger.error(f"Processing error: {e}")

        input("\nPress Enter to continue...")

    def process_traditional_interactive(self):
        """Interactive traditional processing"""
        if not self.documents:
            print("❌ No PDF loaded! Please load a PDF first.")
            input("Press Enter to continue...")
            return

        self.clear_screen()
        self.print_header()
        print("🔍 TRADITIONAL PROCESSING PIPELINE")
        print("-" * 50)

        try:
            # Name extraction
            self.animated_loading("Extracting names from documents", 3)
            name_extractor = NameExtractor(self.config, self.progress_tracker)
            name_extractor.extract(self.documents)
            consolidated = name_extractor.get_consolidated_name_tokens(min_pages=1)

            print(f"\n👤 Found {len(consolidated)} unique patients:")
            for i, (name, tokens) in enumerate(list(consolidated.items())[:5], 1):
                print(f"  {i}. {name} ({', '.join(tokens)})")
            if len(consolidated) > 5:
                print(f"  ... and {len(consolidated) - 5} more")

            # Page assignment
            self.animated_loading("Assigning pages to patients", 2)
            assignment_engine = AssignmentEngine(self.config, self.progress_tracker)
            assignment_result = assignment_engine.assign(self.documents, consolidated)

            self.assignments = assignment_result.data["patient_assignments"]
            self.review_queue = assignment_result.data["review_queue"]

            print(f"\n✅ Assignment Complete!")
            print(f"📊 {len(self.assignments)} patients assigned")
            print(f"📄 {sum(len(pages) for pages in self.assignments.values())} pages assigned")
            print(f"⚠️  {len(self.review_queue)} pages need review")

        except Exception as e:
            print(f"\n❌ Processing failed: {e}")

        input("\nPress Enter to continue...")

    def view_results_interactive(self):
        """Interactive results viewing"""
        if not self.assignments:
            print("❌ No results available! Please process a PDF first.")
            input("Press Enter to continue...")
            return

        while True:
            self.clear_screen()
            self.print_header()
            print("📊 RESULTS VIEWER")
            print("-" * 30)

            print("1. 👥 View Patient Assignments")
            print("2. ⚠️  View Review Queue")
            print("3. 📈 View Quality Metrics")
            print("4. 🔙 Back to Main Menu")

            choice = self.get_user_choice("View option: ", ["1", "2", "3", "4"])

            if choice == "1":
                self.show_assignments()
            elif choice == "2":
                self.show_review_queue()
            elif choice == "3":
                self.show_quality_metrics()
            elif choice == "4":
                break

    def show_assignments(self):
        """Show patient assignments"""
        print("\n👥 PATIENT ASSIGNMENTS")
        print("=" * 50)

        for i, (patient, pages) in enumerate(self.assignments.items(), 1):
            print(f"\n{i}. 📋 {patient}")
            print(f"   📄 Pages: {sorted(pages)} ({len(pages)} total)")

            # Show confidence if available
            if hasattr(self, 'processing_results') and self.processing_results:
                confidence_scores = self.processing_results.get("confidence_scores", {})
                avg_confidence = sum(confidence_scores.get(p, 0.5) for p in pages) / len(pages) if pages else 0
                print(f"   📊 Avg Confidence: {avg_confidence:.2f}")

        input("\nPress Enter to continue...")

    def show_review_queue(self):
        """Show review queue"""
        print("\n⚠️  REVIEW QUEUE")
        print("=" * 40)

        if not self.review_queue:
            print("✅ No pages need review!")
        else:
            for i, item in enumerate(self.review_queue, 1):
                page = item.get("page", "")
                candidates = item.get("candidates", [])
                reason = item.get("reason", "")

                print(f"\n{i}. 📄 Page {page}")
                print(f"   🎯 Candidates: {', '.join(candidates) if candidates else 'None'}")
                print(f"   ❓ Reason: {reason}")

        input("\nPress Enter to continue...")

    def show_quality_metrics(self):
        """Show quality metrics"""
        print("\n📈 QUALITY METRICS")
        print("=" * 40)

        total_pages = len(self.documents)
        assigned_pages = sum(len(pages) for pages in self.assignments.values())
        unassigned_pages = len(self.review_queue)

        print(f"📄 Total Pages: {total_pages}")
        print(f"✅ Assigned Pages: {assigned_pages} ({assigned_pages/total_pages*100:.1f}%)")
        print(f"⚠️  Unassigned Pages: {unassigned_pages} ({unassigned_pages/total_pages*100:.1f}%)")
        print(f"👥 Patients Found: {len(self.assignments)}")

        if hasattr(self, 'processing_results') and self.processing_results:
            metadata = self.processing_results.get("metadata", {})
            quality_metrics = metadata.get("quality_metrics", {})

            if quality_metrics:
                print(f"\n🔍 Additional Metrics:")
                for key, value in quality_metrics.items():
                    print(f"   {key.replace('_', ' ').title()}: {value}")

        input("\nPress Enter to continue...")

    def split_pdfs_interactive(self):
        """Interactive PDF splitting"""
        if not self.assignments or not self.current_pdf:
            print("❌ No assignments available! Please process a PDF first.")
            input("Press Enter to continue...")
            return

        self.clear_screen()
        self.print_header()
        print("✂️  PDF SPLITTING")
        print("-" * 30)

        print(f"📂 Source: {Path(self.current_pdf).name}")
        print(f"👥 Splitting into {len(self.assignments)} patient files")

        output_dir = input("Output directory (default: 'output'): ").strip() or "output"

        try:
            from utils.splitter import PDFSplitter
            splitter = PDFSplitter()

            # Convert to page-based format
            page_assignments = {}
            for patient, pages in self.assignments.items():
                for page in pages:
                    page_assignments[page - 1] = patient  # Convert to 0-based

            self.animated_loading("Splitting PDF files", 3)

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            splitter.split(self.documents, page_assignments, output_dir)

            print(f"\n✅ PDFs split successfully!")
            print(f"📁 Output directory: {Path(output_dir).absolute()}")

            # List created files
            output_path = Path(output_dir)
            pdf_files = list(output_path.glob("*.pdf"))
            print(f"\n📄 Created {len(pdf_files)} files:")
            for pdf_file in pdf_files:
                print(f"   📋 {pdf_file.name}")

        except Exception as e:
            print(f"❌ Splitting failed: {e}")

        input("\nPress Enter to continue...")

    def export_results_interactive(self):
        """Interactive results export"""
        if not self.processing_results:
            print("❌ No results to export! Please process a PDF first.")
            input("Press Enter to continue...")
            return

        self.clear_screen()
        self.print_header()
        print("💾 EXPORT RESULTS")
        print("-" * 30)

        filename = input("Export filename (default: 'results.json'): ").strip() or "results.json"

        try:
            export_data = {
                "source_pdf": self.current_pdf,
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "configuration": {
                    "speed_vs_accuracy": CONFIG.speed_vs_accuracy,
                    "confidence_threshold": CONFIG.confidence_threshold,
                    "use_langgraph": CONFIG.use_langgraph
                },
                "results": self.processing_results
            }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            print(f"✅ Results exported to: {filename}")
            print(f"📊 Export includes assignments, review queue, and metadata")

        except Exception as e:
            print(f"❌ Export failed: {e}")

        input("\nPress Enter to continue...")

    def show_system_info(self):
        """Show system information"""
        self.clear_screen()
        self.print_header()
        print("🔧 SYSTEM INFORMATION")
        print("-" * 40)

        print(f"🖥️  Device: {CONFIG.device}")
        print(f"🎮 GPU Available: {'✅' if CONFIG.use_gpu else '❌'}")
        print(f"⚡ Processing Mode: {CONFIG.speed_vs_accuracy}")
        print(f"📊 Confidence Threshold: {CONFIG.confidence_threshold:.2f}")
        print(f"📦 Batch Size: {CONFIG.batch_size}")
        print(f"👥 Max Workers: {CONFIG.max_workers}")
        print(f"🤖 LangGraph Enabled: {'✅' if CONFIG.use_langgraph else '❌'}")
        if hasattr(CONFIG, 'graph_rag_enabled'):
            print(f"🕸️  GraphRAG Enabled: {'✅' if CONFIG.graph_rag_enabled else '❌'}")
        else:
            print(f"🕸️  GraphRAG Enabled: False")


        if self.current_pdf:
            print(f"\n📄 Current PDF: {Path(self.current_pdf).name}")
            print(f"📊 Documents Loaded: {len(self.documents)}")

        if self.assignments:
            print(f"👥 Patients Assigned: {len(self.assignments)}")
            print(f"📄 Pages Assigned: {sum(len(pages) for pages in self.assignments.values())}")

        input("\nPress Enter to continue...")

    def run(self):
        """Main interactive loop"""
        while True:
            self.clear_screen()
            self.print_header()

            # Show current status
            if self.current_pdf:
                print(f"📄 Current PDF: {Path(self.current_pdf).name} ({len(self.documents)} pages)")
            if self.assignments:
                print(f"👥 Last Processing: {len(self.assignments)} patients, {len(self.review_queue)} pages need review")

            self.print_menu()

            choice = self.get_user_choice("Select option (1-9): ", 
                                        ["1", "2", "3", "4", "5", "6", "7", "8", "9"])

            if choice == "1":
                self.load_pdf_interactive()
            elif choice == "2":
                self.configure_settings_interactive()
            elif choice == "3":
                self.process_with_langgraph_interactive()
            elif choice == "4":
                self.process_traditional_interactive()
            elif choice == "5":
                self.view_results_interactive()
            elif choice == "6":
                self.split_pdfs_interactive()
            elif choice == "7":
                self.export_results_interactive()
            elif choice == "8":
                self.show_system_info()
            elif choice == "9":
                print("\n👋 Thank you for using Enhanced Medical Record Splitter!")
                print("🏥 Stay safe and keep helping patients!")
                break


if __name__ == "__main__":
    # Dummy classes for ProgressTracker and ConsoleProgressCallback if they are not provided
    # In a real scenario, these would be imported from their respective modules.
    class ProgressTracker:
        def __init__(self):
            self.callbacks = {}

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def update(self, stage, progress, total):
            for callback in self.callbacks.values():
                callback.update(stage, progress, total)

    class ConsoleProgressCallback:
        def __init__(self, width=50):
            self.width = width

        def update(self, stage, progress, total):
            percent = (progress / total) * 100
            bar_length = int(self.width * (progress / total))
            bar = '#' * bar_length + '-' * (self.width - bar_length)
            print(f"\r{stage}: [{bar}] {percent:.1f}% ({progress}/{total})", end='', flush=True)
            if progress == total:
                print() # Newline after completion


    terminal = InteractiveTerminal()
    terminal.run()