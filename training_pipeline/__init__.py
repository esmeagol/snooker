"""
Snooker Training Data Pipeline

A unified pipeline for generating training data for snooker object detection models.
"""

__version__ = "0.1.0"

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
