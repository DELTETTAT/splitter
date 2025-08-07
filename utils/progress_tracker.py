"""
Comprehensive progress tracking system
"""
import time
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from core.base import ProcessingStage


@dataclass
class ProgressUpdate:
    """Progress update data structure"""
    stage: ProcessingStage
    progress: float  # 0.0 to 1.0
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Enhanced progress tracking with callbacks and history"""

    def __init__(self):
        self.callbacks: Dict[str, Callable] = {}
        self.history: list[ProgressUpdate] = []
        self.current_stage: Optional[ProcessingStage] = None
        self.stage_start_times: Dict[ProcessingStage, float] = {}
        self.total_start_time: Optional[float] = None

    def add_callback(self, name: str, callback: Callable):
        """Add a progress callback"""
        self.callbacks[name] = callback

    def remove_callback(self, name: str):
        """Remove a progress callback"""
        if name in self.callbacks:
            del self.callbacks[name]

    def update(self, stage: ProcessingStage, progress: float, 
               message: str = "", metadata: Dict[str, Any] = None):
        """Update progress and notify all callbacks"""

        # Initialize timing if needed
        if self.total_start_time is None:
            self.total_start_time = time.time()

        # Track stage timing
        if stage != self.current_stage:
            if stage not in self.stage_start_times:
                self.stage_start_times[stage] = time.time()
            self.current_stage = stage

        # Create progress update
        update = ProgressUpdate(
            stage=stage,
            progress=progress,
            message=message,
            metadata=metadata or {}
        )

        # Add timing information
        if self.total_start_time:
            update.metadata["elapsed_total"] = time.time() - self.total_start_time

        if stage in self.stage_start_times:
            update.metadata["elapsed_stage"] = time.time() - self.stage_start_times[stage]

        # Store in history
        self.history.append(update)

        # Notify callbacks
        for callback in self.callbacks.values():
            try:
                callback(update)
            except Exception as e:
                print(f"Progress callback error: {e}")

    def get_stage_summary(self) -> Dict[ProcessingStage, Dict[str, Any]]:
        """Get summary of all processing stages"""
        summary = {}

        for stage in ProcessingStage:
            stage_updates = [u for u in self.history if u.stage == stage]
            if stage_updates:
                first_update = stage_updates[0]
                last_update = stage_updates[-1]

                summary[stage] = {
                    "started": first_update.timestamp,
                    "completed": last_update.timestamp if last_update.progress >= 1.0 else None,
                    "duration": last_update.timestamp - first_update.timestamp,
                    "final_progress": last_update.progress,
                    "final_message": last_update.message,
                    "update_count": len(stage_updates)
                }

        return summary

    def get_overall_progress(self) -> float:
        """Calculate overall progress across all stages"""
        if not self.history:
            return 0.0

        # Weight each stage equally
        stage_weights = {
            ProcessingStage.INITIALIZATION: 0.05,
            ProcessingStage.TEXT_EXTRACTION: 0.15,
            ProcessingStage.NAME_EXTRACTION: 0.15,
            ProcessingStage.CLASSIFICATION: 0.10,
            ProcessingStage.ENTITY_EXTRACTION: 0.15,
            ProcessingStage.PATIENT_ASSIGNMENT: 0.20,
            ProcessingStage.VALIDATION: 0.10,
            ProcessingStage.QUALITY_CHECK: 0.05,
            ProcessingStage.SUMMARY_GENERATION: 0.05
        }

        overall_progress = 0.0

        for stage, weight in stage_weights.items():
            stage_updates = [u for u in self.history if u.stage == stage]
            if stage_updates:
                latest_progress = stage_updates[-1].progress
                overall_progress += weight * latest_progress

        return min(overall_progress, 1.0)

    def reset(self):
        """Reset progress tracking"""
        self.history.clear()
        self.current_stage = None
        self.stage_start_times.clear()
        self.total_start_time = None


class ConsoleProgressCallback:
    """Console-based progress display"""

    def __init__(self, width: int = 50):
        self.width = width
        self.last_stage = None

    def __call__(self, update):
        """Display progress update in console"""
        # Handle both ProgressUpdate objects and direct calls
        if hasattr(update, 'stage'):
            stage = update.stage
            progress = update.progress
            message = update.message
            metadata = update.metadata
        else:
            # Handle direct callback from components
            stage = update
            progress = 0.5
            message = "Processing..."
            metadata = {}
        
        # Show stage header if changed
        stage_name = stage.value if hasattr(stage, 'value') else str(stage)
        if stage != self.last_stage:
            print(f"\n🔄 {stage_name.replace('_', ' ').title()}")
            self.last_stage = stage

        # Create progress bar
        filled = int(progress * self.width)
        bar = "█" * filled + "░" * (self.width - filled)
        percentage = progress * 100

        # Show progress line
        elapsed = metadata.get("elapsed_stage", 0) if isinstance(metadata, dict) else 0
        print(f"\r[{bar}] {percentage:5.1f}% - {message} ({elapsed:.1f}s)", 
              end="", flush=True)

        if progress >= 1.0:
            print()  # New line when complete


class DetailedProgressCallback:
    """Detailed progress callback for GUI applications"""

    def __init__(self, update_ui_func: Callable):
        self.update_ui_func = update_ui_func
        self.stage_history = {}

    def __call__(self, update: ProgressUpdate):
        """Update GUI with detailed progress information"""
        # Store stage history
        if update.stage not in self.stage_history:
            self.stage_history[update.stage] = []
        self.stage_history[update.stage].append(update)

        # Prepare data for UI update
        ui_data = {
            "current_stage": update.stage.value,
            "stage_progress": update.progress,
            "stage_message": update.message,
            "elapsed_total": update.metadata.get("elapsed_total", 0),
            "elapsed_stage": update.metadata.get("elapsed_stage", 0),
            "timestamp": update.timestamp,
            "stage_history": self.stage_history
        }

        # Update UI
        try:
            self.update_ui_func(ui_data)
        except Exception as e:
            print(f"UI update failed: {e}")