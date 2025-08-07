import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class Config:
    """Configuration for AI models and processing parameters"""

    # Device configuration
    device: str = field(default_factory=lambda: "cuda" if TORCH_AVAILABLE and
                        torch.cuda.is_available() else "cpu")
    use_gpu: bool = field(
        default_factory=lambda: TORCH_AVAILABLE and torch.cuda.is_available())

    # Model paths (for local deployment)
    local_llm_path: Optional[str] = None
    local_ner_model: str = "dslim/bert-large-NER"

    # Processing parameters
    speed_vs_accuracy: str = "balanced"  # "fast", "balanced", "accurate"
    confidence_threshold: float = 0.7
    ocr_confidence_threshold: int = 70

    # Batch processing
    batch_size: int = 4
    max_workers: int = 4

    # Fine-tuning parameters
    enable_fine_tuning: bool = False
    learning_rate: float = 2e-5

    # LangGraph settings
    use_langgraph: bool = True
    graph_rag_enabled: bool = False

    # Output configuration
    output_dir: str = "output"
    log_level: int = field(default_factory=lambda: logging.INFO)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration based on speed vs accuracy setting"""
        configs = {
            "fast": {
                "model_size": "small",
                "batch_size": 8,
                "confidence_threshold": 0.6,
                "ocr_pages_parallel": True
            },
            "balanced": {
                "model_size": "medium",
                "batch_size": 4,
                "confidence_threshold": 0.7,
                "ocr_pages_parallel": True
            },
            "accurate": {
                "model_size": "large",
                "batch_size": 2,
                "confidence_threshold": 0.8,
                "ocr_pages_parallel": False
            }
        }
        return configs.get(self.speed_vs_accuracy, configs["balanced"])

    def update_from_dict(self, params: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __dict__(self):
        """Return configuration as dictionary"""
        return {
            "device": self.device,
            "use_gpu": self.use_gpu,
            "speed_vs_accuracy": self.speed_vs_accuracy,
            "confidence_threshold": self.confidence_threshold,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "use_langgraph": self.use_langgraph,
            "graph_rag_enabled": self.graph_rag_enabled,
            "ocr_confidence_threshold": self.ocr_confidence_threshold,
            "output_dir": self.output_dir,
            "log_level": self.log_level
        }


# Global configuration instance
CONFIG = Config()