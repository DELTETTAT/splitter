
import argparse
import logging
from pathlib import Path
from extractor import PDFExtractor
from local_ai_engine import AI_ENGINE
from langgraph_processor import LANGGRAPH_PROCESSOR
from config import CONFIG
from enhanced_ui import EnhancedReviewUI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cli(pdf_path: str, output_dir: str = "output", use_langgraph: bool = True):
    """Run processing via command line interface"""
    try:
        # Load and configure
        logger.info(f"Processing {pdf_path} with config: {CONFIG.__dict__}")
        
        # Extract documents
        extractor = PDFExtractor(
            ocr_confidence_threshold=CONFIG.ocr_confidence_threshold,
            num_workers=CONFIG.max_workers
        )
        documents = extractor.text_extractor(pdf_path)
        logger.info(f"Extracted {len(documents)} documents")
        
        # Process with LangGraph or traditional pipeline
        if use_langgraph and CONFIG.use_langgraph:
            logger.info("Using LangGraph processing pipeline")
            result_state = LANGGRAPH_PROCESSOR.process_documents(documents)
            
            assignments = result_state.get("patient_assignments", {})
            review_queue = result_state.get("review_queue", [])
            quality_metrics = result_state.get("metadata", {}).get("quality_metrics", {})
            
        else:
            logger.info("Using traditional processing pipeline")
            from assigner import assign_pages_to_patients
            from name_extractor import NameExtractor
            
            name_extractor = NameExtractor()
            name_extractor.extract(documents)
            consolidated = name_extractor.get_consolidated_name_tokens(min_pages=1)
            
            assignments, review = assign_pages_to_patients(documents, consolidated)
            review_queue = [
                {"page": page, "candidates": candidates, "reason": reason}
                for page, candidates, reason in review
            ]
            quality_metrics = {
                "total_pages": len(documents),
                "assigned_pages": sum(len(pages) for pages in assignments.values()),
                "unassigned_pages": len(review_queue),
                "patients_identified": len(assignments)
            }
        
        # Print results
        print(f"\n🏥 Processing Results for {Path(pdf_path).name}")
        print("=" * 50)
        
        print(f"\n📊 Quality Metrics:")
        for key, value in quality_metrics.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n👥 Patient Assignments ({len(assignments)} patients):")
        for patient, pages in assignments.items():
            print(f"  📋 {patient}: {len(pages)} pages ({sorted(pages)})")
        
        if review_queue:
            print(f"\n⚠️  Pages Requiring Review ({len(review_queue)} pages):")
            for item in review_queue[:10]:  # Show first 10
                page = item.get("page", "")
                reason = item.get("reason", "")
                print(f"  📄 Page {page}: {reason}")
            if len(review_queue) > 10:
                print(f"  ... and {len(review_queue) - 10} more pages")
        
        # Split PDFs if requested
        if output_dir:
            from splitter import PDFSplitter
            splitter = PDFSplitter()
            
            # Convert assignments to page-based format
            page_assignments = {}
            for patient, pages in assignments.items():
                for page in pages:
                    page_assignments[page - 1] = patient  # Convert to 0-based
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            splitter.split(documents, page_assignments, output_dir)
            print(f"\n💾 PDFs split and saved to: {output_dir}")
        
        return assignments, review_queue, quality_metrics
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

def run_gui():
    """Run the enhanced GUI interface"""
    logger.info("Starting enhanced GUI")
    app = EnhancedReviewUI()
    app.mainloop()

def run_terminal():
    """Run the interactive terminal interface"""
    logger.info("Starting interactive terminal")
    from interactive_terminal import InteractiveTerminal
    terminal = InteractiveTerminal()
    terminal.run()

def main():
    parser = argparse.ArgumentParser(description="Enhanced Medical Record PDF Splitter")
    parser.add_argument("--pdf", help="Path to PDF file to process")
    parser.add_argument("--output", default="output", help="Output directory for split PDFs")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    parser.add_argument("--terminal", action="store_true", help="Launch interactive terminal interface")
    parser.add_argument("--no-langgraph", action="store_true", help="Disable LangGraph processing")
    
    # Configuration arguments
    parser.add_argument("--speed", choices=["fast", "balanced", "accurate"], 
                       default="balanced", help="Speed vs accuracy trade-off")
    parser.add_argument("--confidence", type=float, default=0.7, 
                       help="Confidence threshold (0.1-1.0)")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Batch size for processing")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Update configuration
    CONFIG.speed_vs_accuracy = args.speed
    CONFIG.confidence_threshold = args.confidence
    CONFIG.batch_size = args.batch_size
    CONFIG.max_workers = args.workers
    CONFIG.use_langgraph = not args.no_langgraph
    
    logger.info(f"🚀 Starting Enhanced Medical Record Splitter")
    logger.info(f"🔧 Configuration: {CONFIG.__dict__}")
    logger.info(f"🖥️  Device: {CONFIG.device} | GPU Available: {CONFIG.use_gpu}")
    
    if args.gui:
        run_gui()
    elif args.terminal:
        run_terminal()
    elif args.pdf:
        run_cli(args.pdf, args.output, not args.no_langgraph)
    else:
        print("Please specify --pdf for CLI mode, --gui for GUI mode, or --terminal for interactive terminal")
        parser.print_help()

if __name__ == "__main__":
    main()
